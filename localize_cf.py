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
    sentence_model_name = 'all-mpnet-base-v2',
    device = 'cuda'):
    global model, preprocessor, sentence_model
    sentence_model = SentenceTransformer(sentence_model_name)
    if model_type == 'owl':
        model = OwlViTModel.from_pretrained(model_name).to(device)
        preprocessor = AutoProcessor.from_pretrained(model_name)
    else:
        model, preprocessor = clip.load(model_name, device=device)
    return model, preprocessor, sentence_model

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
    min_coords = None,
    device = 'cuda'
):
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
    ).to(device)
    model_weights = torch.load(model_weights_path, map_location=device)
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
    o3d.io.write_point_cloud(f"point_cloud.ply", merged_downpcd)
    return points_dataloader

def calculate_clip_and_st_embeddings_for_queries(queries, model, preprocessor,
         sentence_model, device = 'cuda', model_type = 'owl'):
    with torch.no_grad():
        if model_type == 'owl':
            inputs = preprocessor(
                text=[queries], return_tensors="pt"
            )
            inputs['input_ids'] = inputs['input_ids'].to(device)
            inputs['attention_mask'] = inputs['attention_mask'].to(device)
            all_clip_tokens = model.get_text_features(**inputs)
        else:
            all_clip_queries = clip.tokenize(queries)
            all_clip_tokens = model.encode_text(all_clip_queries.to(device)).float()
        all_clip_tokens = F.normalize(all_clip_tokens, p=2, dim=-1)
        all_st_tokens = torch.from_numpy(sentence_model.encode(queries))
        all_st_tokens = F.normalize(all_st_tokens, p=2, dim=-1).to(device)
    return all_clip_tokens, all_st_tokens

def find_alignment_over_model(label_model, clip_model, preprocessor, sentence_model, queries, dataloader,
            vision_weight = 10.0, text_weight = 10.0, linguistic = 'owl', device = 'cuda'):
    clip_text_tokens, st_text_tokens = calculate_clip_and_st_embeddings_for_queries(
            queries, clip_model, preprocessor, sentence_model, model_type = linguistic)
    point_alignments = []
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader, total = len(dataloader)):
            # Find alignmnents with the vectors
            predicted_label_latents, predicted_image_latents = label_model(data.to(device))
            data_text_tokens = F.normalize(predicted_label_latents, p=2, dim=-1).to(device)
            text_alignment = data_text_tokens @ st_text_tokens.T
            data_visual_tokens = F.normalize(predicted_image_latents, p=2, dim=-1).to(device)
            visual_alignment = data_visual_tokens @ clip_text_tokens.T
            total_alignment = (text_weight * text_alignment) + (vision_weight * visual_alignment)
            total_alignment /= (text_weight + vision_weight)
            point_alignments.append(total_alignment)

    point_alignments = torch.cat(point_alignments).T
    print(point_alignments.shape)
    return point_alignments

# Currently we only support compute one query each time, in the future we might want to support check many queries

def localize_AonB(label_model, clip_model, preprocessor, sentence_model, A, B, dataloader, k_A = 10, k_B = 1000,
        vision_weight = 10.0, text_weight = 10.0, linguistic = 'clip'):
    if B is None or B == '':
        target = find_alignment_for_A(label_model, clip_model, preprocessor, sentence_model, [A], dataloader,
            vision_weight = vision_weight, text_weight = text_weight, linguistic = linguistic)[0]
    else:
        alignments = find_alignment_over_model(label_model, clip_model, preprocessor, sentence_model,
                [A, B], dataloader, vision_weight = vision_weight, text_weight = text_weight, linguistic = linguistic).cpu()
        A_points = dataloader.dataset[alignments.topk(k = k_A, dim = -1).indices].reshape(-1, 3)
        B_points = dataloader.dataset[alignments.topk(k = k_B, dim = -1).indices].reshape(-1, 3)
        distances = torch.norm(A_points.unsqueeze(1) - B_points.unsqueeze(0), dim=2)
        target = A_points[torch.argmin(torch.min(distances, dim = 0).values)]
    target = target[[0, 2, 1]]
    target[1] = -target[1]
    return target

# Keep in mind that the objective is picking up A from B
def find_alignment_for_B(label_model, clip_model, preprocessor, sentence_model, B, dataloader, 
            # for this function only, how many points do you want to considered as relevant to B
            threshold_precentile = 0.01,
            # for clip-fields
            vision_weight = 10.0, text_weight = 10.0, linguistic = 'owl'):
    assert threshold_precentile > 0 and threshold_precentile <= 1, 'invalid threshold_precentile'
    alignments = find_alignment_over_model(label_model, clip_model, preprocessor, sentence_model, 
                B, dataloader, vision_weight, text_weight, linguistic).cpu()
    threshold = int(len(dataloader.dataset) * threshold_precentile)
    B_dataset = dataloader.dataset[alignments.topk(k = threshold, dim = -1).indices]
    BATCH_SIZE = 30_000
    B_dataset = B_dataset.reshape(-1, 3)
    return DataLoader(B_dataset, batch_size = BATCH_SIZE, num_workers = 10)

def find_alignment_for_A(label_model, clip_model, preprocessor, sentence_model, A, dataloader, 
            # for clip-fields
            vision_weight = 10.0, text_weight = 10.0, linguistic = 'owl'):
    alignments = find_alignment_over_model(label_model, clip_model, preprocessor, sentence_model, 
            A, dataloader, vision_weight, text_weight, linguistic).cpu()
    return dataloader.dataset[alignments.argmax(dim = -1)]
        

# Note that even though we cannot localize many queries with LERF at the same time,
# we can actually select lerf scales for many queries at the same time.

def load_everything(config_path):
    config = OmegaConf.load(config_path)
    model_type = config.web_models.segmentation
    dataset_path = os.path.join('clip-fields', config.saved_dataset_path)
    if model_type != 'owl':
        model_name = 'ViT-B/32'
    else:
        model_name = 'google/owlvit-base-patch32'
    weight_path = os.path.join('clip-fields', config.save_directory, 'implicit_scene_label_model_latest.pt')
    clip_model, preprocessor, sentence_model = load_pretrained(model_type = model_type , model_name = model_name)
    points_dataloader = get_dataloader(cf_path = dataset_path)
    max_coords, _ = points_dataloader.dataset.max(dim=0)
    min_coords, _ = points_dataloader.dataset.min(dim=0)
    label_model = load_field(
        config_path = config_path, model_weights_path = weight_path, max_coords = max_coords, min_coords = min_coords)
    return label_model, clip_model, preprocessor, sentence_model, points_dataloader, model_type

def localize_pickupAfromB(label_model, clip_model, preprocessor, sentence_model, points_dataloader, model_type,
             A, B, config_path, vision_weight = 10.0, text_weight = 10.0):
    B_dataloader = find_alignment_for_B(
        label_model, clip_model, preprocessor, sentence_model, [B], points_dataloader, 
        linguistic = model_type, vision_weight = vision_weight, text_weight = text_weight)
    final_point = find_alignment_for_A(
        label_model, clip_model, preprocessor, sentence_model, [A], B_dataloader, 
        linguistic = model_type, vision_weight = vision_weight, text_weight = text_weight)
    del B_dataloader
    final_point[:, -1] = -final_point[:, -1]
    return final_point[0, [0, 2, 1]]

def evaluate():
    label_model, clip_model, preprocessor, sentence_model, points_dataloader, model_type = load_everything('clip-fields/configs/train.yaml')

    eval_data = pd.read_csv('clip-fields/kitchen.csv')
    queries = list(eval_data['query'])

    xs, ys, zs, affords = list(eval_data['x']), list(eval_data['y']), list(eval_data['z']), list(eval_data['affordance'])
    xyzs = torch.stack([torch.tensor(xs), torch.tensor(ys), torch.tensor(zs)], dim = 1)
    max_points = find_alignment_for_A(label_model, clip_model, preprocessor, sentence_model, 
            queries, points_dataloader,
            vision_weight = 10.0, text_weight = 10.0, linguistic = model_type)
    print(max_points.shape)
    for max_point, query in zip(max_points, queries):
        print(max_point, query)

    correctness = torch.linalg.norm((max_points[:, [0, 2]] - xyzs[:, [0, 2]]), dim = -1) <= torch.tensor(affords)
    print(np.array(queries)[torch.where(correctness)[0].numpy()], 
        np.array(queries)[torch.where(~correctness)[0].numpy()], 
        len(np.array(queries)[torch.where(correctness)[0].numpy()]) / len(correctness))

#label_model, clip_model, preprocessor, sentence_model, points_dataloader, model_type = load_everything('clip-fields/configs/train.yaml')
#print(localize_AonB(label_model, clip_model, preprocessor, sentence_model, 'plant', '', points_dataloader, k_A = 10, k_B = 1000, linguistic = model_type))
#evaluate()
