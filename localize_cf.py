import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain, cycle
from sentence_transformers import SentenceTransformer, util

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import tqdm
import einops

import os
import sys
sys.path.append('clip-fields')
sys.path.append('usa')
sys.path.append('lerf')

from dataloaders.real_dataset import DeticDenseLabelledDataset
from grid_hash_model import GridCLIPModel

from misc import MLP

import pandas as pd
import pyntcloud
from pyntcloud import PyntCloud
import clip
import pandas as pd

from transformers import AutoProcessor, OwlViTModel

DEVICE = "cuda"

os.environ["TOKENIZERS_PARALLELISM"] = '(true | false)'

from transformers import AutoProcessor, OwlViTModel
from omegaconf import OmegaConf

import ml.api as ml
import usa
from usa.tasks.datasets.posed_rgbd import get_posed_rgbd_dataset, iter_xyz, get_bounds, Bounds
from usa.models.point2emb import Point2EmbModel, Point2EmbModelConfig
from usa.planners.base import get_ground_truth_map_from_dataset
from usa.tasks.clip_sdf import ClipSdfTask

# These environment variables control where training and eval logs are written.
# You can set these in your shell profile as well.
os.environ["RUN_DIR"] = "runs"
os.environ["EVAL_RUN_DIR"] = "eval_runs"
os.environ["MODEL_DIR"] = "models"
os.environ["DATA_DIR"] = "data"

# This is used to set a constant Tensorboard port.
os.environ["TENSORBOARD_PORT"] = str(8989)

# Useful for debugging.
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from lerf.lerf import LERFModelConfig, LERFModel
from lerf.lerf_field import LERFField
import open_clip

torch.manual_seed(10000)

model = None
preprocessor = None
sentence_model = None
bounds = None

# model type can be owl, openclip (only openclip requires weight_name), or clip
# if model_type is 'owl', then default model_name is 'google/owlvit-base-patch32'
# if model_type is 'openclip' or clip, then default model_name is 'ViT-B/32'
def load_pretrained(
    model_type = 'owl', 
    model_name = 'google/owlvit-base-patch32',
    sentence_model_name = 'all-mpnet-base-v2'):
    global model, preprocessor, sentence_model, DEVICE
    sentence_model = SentenceTransformer(sentence_model_name)
    if model_type == 'owl':
        model = OwlViTModel.from_pretrained(model_name).to(DEVICE)
        preprocessor = AutoProcessor.from_pretrained(model_name)
    else:
        model, preprocess = clip.load(model_name, device=DEVICE)

# For this part, if field type is clip-fields or lerf, then, You need to provide your model_weights_path
# if your field type is usa-net, we will load your training config
# If you are using clip VIT-B/32, image rep size is 512
# For label rep, we are only considering SentenceTransformer("all-mpnet-base-v2"), so the label_rep_size is 768
# If you are loading lerf, then feel free to set config_path None
def load_field( 
    model_weights_path, 
    config_path = None, 
    image_rep_size = 512, 
    text_rep_size=768,
    max_coords = None,
    min_coords = None
):
    global DEVICE, lerf_model
    if config_path:
        config = OmegaConf.load(config_path)
    model = GridCLIPModel(
        image_rep_size = image_rep_size,
        text_rep_size = text_rep_size,
        mlp_depth = config.mlp_depth,
        mlp_width = config.mlp_width,
        log2_hashmap_size = config.log2_hashmap_size,
        num_levels = config.num_grid_levels,
        level_dim = config.level_dim,
        per_level_scale = config.per_level_scale,
        max_coords = max_coords,
        min_coords = min_coords
    ).to(DEVICE)
    model_weights = torch.load(model_weights_path, map_location=DEVICE)
    model.load_state_dict(model_weights["model"])
    return model
    

def get_dataloader(cf_path):
    # For this part, you can either choose to load clip fields training data
    # or a r3d file to get an USA-Net styled data
    # bounds is needed to test LERF, so if you choose to load dataloader from cf_path, you should not
    # be able to test LERF
    training_data = torch.load(cf_path)
    data_xyzs = training_data._label_xyz
    batch_size = 30_000
    points_dataloader = DataLoader(
        data_xyzs.detach().cpu(), batch_size=batch_size, num_workers=10,
    )
    print("Created data loader", points_dataloader)
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(points_dataloader.dataset)
    merged_downpcd = merged_pcd.voxel_down_sample(voxel_size=0.03)
    o3d.io.write_point_cloud(f"nyu_lab.ply", merged_downpcd)
    return points_dataloader

def calculate_clip_and_st_embeddings_for_queries(queries, model_type = 'owl'):
    global model, preprocessor, DEVICE
    with torch.no_grad():
        if model_type == 'owl':
            inputs = preprocessor(
                text=[queries], return_tensors="pt"
            )
            inputs['input_ids'] = inputs['input_ids'].to(DEVICE)
            inputs['attention_mask'] = inputs['attention_mask'].to(DEVICE)
            all_clip_tokens = model.get_text_features(**inputs)
        else:
            all_clip_queries = clip.tokenize(queries)
            all_clip_tokens = model.encode_text(all_clip_queries.to(DEVICE)).float()
        all_clip_tokens = F.normalize(all_clip_tokens, p=2, dim=-1)
        all_st_tokens = torch.from_numpy(sentence_model.encode(queries))
        all_st_tokens = F.normalize(all_st_tokens, p=2, dim=-1).to(DEVICE)
    return all_clip_tokens, all_st_tokens

def find_alignment_over_model(label_model, queries, dataloader, 
            # for clip-fields
            vision_weight = 10.0, text_weight = 10.0, linguistic = 'owl'):
    global lerf_model, DEVICE, bounds
    clip_text_tokens, st_text_tokens = calculate_clip_and_st_embeddings_for_queries(
        queries, linguistic)
    point_alignments = []
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader, total = len(dataloader)):
            # Find alignmnents with the vectors
            predicted_label_latents, predicted_image_latents = label_model(data.to(DEVICE))
            data_text_tokens = F.normalize(predicted_label_latents, p=2, dim=-1).to(DEVICE)
            text_alignment = data_text_tokens @ st_text_tokens.T
            data_visual_tokens = F.normalize(predicted_image_latents, p=2, dim=-1).to(DEVICE)
            visual_alignment = data_visual_tokens @ clip_text_tokens.T
            total_alignment = (text_weight * text_alignment) + (vision_weight * visual_alignment)
            total_alignment /= (text_weight + vision_weight)
            point_alignments.append(total_alignment)

    point_alignments = torch.cat(point_alignments).T
    print(point_alignments.shape)
    return point_alignments

# Currently we only support compute one query each time, in the future we might want to support check many queries

# Keep in mind that the objective is picking up A from B
def find_alignment_for_B(label_model, B, dataloader, 
            # for this function only, how many points do you want to considered as relevant to B
            threshold_precentile = 0.05,
            # for clip-fields
            vision_weight = 10.0, text_weight = 10.0, linguistic = 'owl'):
    assert threshold_precentile > 0 and threshold_precentile <= 1, 'invalid threshold_precentile'
    alignments = find_alignment_over_model(label_model, B, dataloader,
                vision_weight, text_weight, linguistic).cpu()
    threshold = int(len(dataloader.dataset) * threshold_precentile)
    B_dataset = dataloader.dataset[alignments.topk(k = threshold, dim = -1).indices]
    BATCH_SIZE = 3000
    B_dataset = B_dataset.reshape(-1, 3)
    return DataLoader(B_dataset, batch_size = BATCH_SIZE, num_workers = 10)

def find_alignment_for_A(label_model, A, dataloader, 
            # for clip-fields
            vision_weight = 10.0, text_weight = 10.0, linguistic = 'owl'):
    alignments = find_alignment_over_model(label_model, A, dataloader,
            vision_weight, text_weight, linguistic).cpu()
    return dataloader.dataset[alignments.argmax(dim = -1)]
        

# Note that even though we cannot localize many queries with LERF at the same time,
# we can actually select lerf scales for many queries at the same time.

MODEL_TYPE = 'owl'
DATASET_PATH = 'clip-fields/detic_labeled_dataset.pt'
if MODEL_TYPE != 'owl':
    MODEL_NAME = 'ViT-B/32'
else:
    MODEL_NAME = 'google/owlvit-base-patch32'
WEIGHT_PATH = 'clip-fields/kitchen_owl1/implicit_scene_label_model_latest.pt'
CONFIG_PATH = 'clip-fields/configs/train.yaml'
load_pretrained(model_type = MODEL_TYPE , model_name = MODEL_NAME)
points_dataloader = get_dataloader(cf_path = DATASET_PATH)
max_coords, _ = points_dataloader.dataset.max(dim=0)
min_coords, _ = points_dataloader.dataset.min(dim=0)
label_model = load_field(config_path = CONFIG_PATH, model_weights_path = WEIGHT_PATH, max_coords = max_coords, min_coords = min_coords)
print(label_model)

eval_data = pd.read_csv('clip-fields/kitchen.csv')
queries = list(eval_data['query'])

xs, ys, zs, affords = list(eval_data['x']), list(eval_data['y']), list(eval_data['z']), list(eval_data['affordance'])
xyzs = torch.stack([torch.tensor(xs), torch.tensor(ys), torch.tensor(zs)], dim = 1)
max_points = find_alignment_for_A(label_model, queries, points_dataloader, 
            vision_weight = 1.0, text_weight = 10.0, linguistic = MODEL_TYPE)
print(max_points.shape)
for max_point, query in zip(max_points, queries):
    print(max_point, query)

correctness = torch.linalg.norm((max_points[:, [0, 2]] - xyzs[:, [0, 2]]), dim = -1) <= torch.tensor(affords)
print(np.array(queries)[torch.where(correctness)[0].numpy()], 
    np.array(queries)[torch.where(~correctness)[0].numpy()], 
    len(np.array(queries)[torch.where(correctness)[0].numpy()]) / len(correctness))
