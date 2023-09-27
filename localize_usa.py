import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain, cycle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import tqdm
import einops

import os
import sys
sys.path.append('usa')

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

# model type can be owl, openclip (only openclip requires weight_name), or clip
# if model_type is 'owl', then default model_name is 'google/owlvit-base-patch32'
# if model_type is 'openclip' or clip, then default model_name is 'ViT-B/32'
def load_pretrained(
    model_type = 'clip', 
    model_name = 'ViT-B/32',
    device = 'cuda'):
    if model_type == 'owl':
        model = OwlViTModel.from_pretrained(model_name).to(device)
        preprocessor = AutoProcessor.from_pretrained(model_name)
    else:
        model, preprocessor = clip.load(model_name, device=device)
    return model, preprocessor

# For this part, if field type is clip-fields or lerf, then, You need to provide your model_weights_path
# if your field type is usa-net, we will load your training config
# If you are using clip VIT-B/32, image rep size is 512
# For label rep, we are only considering SentenceTransformer("all-mpnet-base-v2"), so the label_rep_size is 768
# If you are loading lerf, then feel free to set config_path None
def load_field( 
    model_weights_path, 
    config_path, 
    device = 'cuda'
):
    config = OmegaConf.load(config_path)
    config = Point2EmbModelConfig(**config.model)
    model = Point2EmbModel(config)
    model = model.to(device)
    model_weights = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(model_weights["model"])
    return model
    

def get_dataloader(r3d_path, sample_prob = 0.01):
    # For this part, you can either choose to load clip fields training data
    # or a r3d file to get an USA-Net styled data
    # bounds is needed to test LERF, so if you choose to load dataloader from cf_path, you should not
    # be able to test LERF
    ds = get_posed_rgbd_dataset(key = 'r3d', path = r3d_path)
    data_xyzs = []
    for xyz, mask_tensor in iter_xyz(ds, 'data'):
        data = xyz[~mask_tensor]
        data = data[torch.randperm(len(data))[:int(len(data) * sample_prob)]]
        data_xyzs.append(data)
    data_xyzs = torch.vstack(data_xyzs)
    batch_size = 30_000
    points_dataloader = DataLoader(
        data_xyzs.detach().cpu(), batch_size=batch_size,
    )
    print("Created data loader", points_dataloader)
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(points_dataloader.dataset)
    merged_downpcd = merged_pcd.voxel_down_sample(voxel_size=0.03)
    o3d.io.write_point_cloud(f"point_cloud.ply", merged_downpcd)
    return points_dataloader

def calculate_clip_embeddings_for_queries(queries, model, preprocessor, device = 'cuda', model_type = 'clip'):
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
    return all_clip_tokens

def find_alignment_over_model(label_model, clip_model, preprocessor, queries, dataloader,
            linguistic = 'clip', device = 'cuda'):
    clip_text_tokens = calculate_clip_embeddings_for_queries(
            queries, clip_model, preprocessor, model_type = linguistic)
    point_alignments = []
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader, total = len(dataloader)):
            # Find alignmnents with the vectors
            predicted_image_latents = label_model(data.to(device))[:, :-1]
            data_visual_tokens = F.normalize(predicted_image_latents, p=2, dim=-1).to(device)
            visual_alignment = data_visual_tokens @ clip_text_tokens.T
            point_alignments.append(visual_alignment)

    point_alignments = torch.cat(point_alignments).T
    print(point_alignments.shape)
    return point_alignments

# Currently we only support compute one query each time, in the future we might want to support check many queries

def localize_AonB(label_model, clip_model, preprocessor, A, B, dataloader, k_A = 10, k_B = 1000, linguistic = 'clip'):
    if B is None or B == '':
        return find_alignment_for_A(label_model, clip_model, preprocessor, [A], dataloader, linguistic = linguistic)[0]
    alignments = find_alignment_over_model(label_model, clip_model, preprocessor, 
                [A, B], dataloader, linguistic = linguistic).cpu()
    A_points = dataloader.dataset[alignments.topk(k = k_A, dim = -1).indices].reshape(-1, 3)
    B_points = dataloader.dataset[alignments.topk(k = k_B, dim = -1).indices].reshape(-1, 3)
    distances = torch.norm(A_points.unsqueeze(1) - B_points.unsqueeze(0), dim=2)
    return A_points[torch.argmin(torch.min(distances, dim = 0).values)]

# Keep in mind that the objective is picking up A from B
def find_alignment_for_B(label_model, clip_model, preprocessor, B, dataloader, 
            # threshold_precentile = how many points do you want to considered as relevant to B
            threshold_precentile = 0.01, linguistic = 'clip'):
    assert threshold_precentile > 0 and threshold_precentile <= 1, 'invalid threshold_precentile'
    alignments = find_alignment_over_model(label_model, clip_model, preprocessor, 
                B, dataloader, linguistic).cpu()
    threshold = int(len(dataloader.dataset) * threshold_precentile)
    B_dataset = dataloader.dataset[alignments.topk(k = threshold, dim = -1).indices]
    BATCH_SIZE = 30_000
    B_dataset = B_dataset.reshape(-1, 3)
    return DataLoader(B_dataset, batch_size = BATCH_SIZE, num_workers = 10)

def find_alignment_for_A(label_model, clip_model, preprocessor, A, dataloader, linguistic = 'clip'):
    alignments = find_alignment_over_model(label_model, clip_model, preprocessor,
            A, dataloader, linguistic).cpu()
    return dataloader.dataset[alignments.argmax(dim = -1)]
        

# Note that even though we cannot localize many queries with LERF at the same time,
# we can actually select lerf scales for many queries at the same time.

def load_everything(config_path, weight_path):
    config = OmegaConf.load(config_path)
    dataset_path = os.path.join('usa', config.task.dataset_path)
    # for now we do not consider owl-vit in USA-Net, yet we strongly believe owl-vit is promising for USA-Net's future
    model_type = "clip"
    if model_type != 'owl':
        model_name = 'ViT-B/32'
    else:
        model_name = 'google/owlvit-base-patch32'
    clip_model, preprocessor = load_pretrained(model_type = model_type , model_name = model_name)
    points_dataloader = get_dataloader(dataset_path)
    max_coords, _ = points_dataloader.dataset.max(dim=0)
    min_coords, _ = points_dataloader.dataset.min(dim=0)
    label_model = load_field(
        config_path = config_path, model_weights_path = weight_path)
    return label_model, clip_model, preprocessor, points_dataloader, model_type

def localize_pickupAfromB(label_model, clip_model, preprocessor, points_dataloader, model_type,
             A, B, config_path):
    B_dataloader = find_alignment_for_B(
        label_model, clip_model, preprocessor, [B], points_dataloader, 
        linguistic = model_type)
    final_point = find_alignment_for_A(
        label_model, clip_model, preprocessor, [A], B_dataloader, 
        linguistic = model_type)
    del B_dataloader
    final_point[:, -1] = -final_point[:, -1]
    return final_point[0, [0, 2, 1]]

def evaluate():
    label_model, clip_model, preprocessor, points_dataloader, model_type = load_everything('usa/configs/train.yaml', 'usa/usa/4_256_no/run_0/checkpoints/ckpt.8000.pt')

    eval_data = pd.read_csv('clip-fields/kitchen.csv')
    queries = list(eval_data['query'])

    xs, ys, zs, affords = list(eval_data['x']), list(eval_data['y']), list(eval_data['z']), list(eval_data['affordance'])
    xyzs = torch.stack([torch.tensor(xs), -torch.tensor(zs), torch.tensor(ys)], dim = 1)
    max_points = find_alignment_for_A(label_model, clip_model, preprocessor, 
            queries, points_dataloader, linguistic = model_type)
    print(max_points.shape)
    for max_point, query in zip(max_points, queries):
        print(max_point, query)

    correctness = torch.linalg.norm((max_points[:, [0, 1]] - xyzs[:, [0, 1]]), dim = -1) <= torch.tensor(affords)
    print(np.array(queries)[torch.where(correctness)[0].numpy()], 
        np.array(queries)[torch.where(~correctness)[0].numpy()], 
        len(np.array(queries)[torch.where(correctness)[0].numpy()]) / len(correctness))

#label_model, clip_model, preprocessor, points_dataloader, model_type = load_everything('usa/configs/train.yaml', 'usa/CDSLab/4_256_no/run_0/checkpoints/ckpt.pt')
#print(localize_AonB(label_model, clip_model, preprocessor, 'trash can', '', points_dataloader, k_A = 10, k_B = 1000, linguistic = model_type))
#evaluate()
#max_points = find_alignment_for_A(label_model, clip_model, preprocessor,
#                 ['bowl', 'green bottle', 'ping pong ball', 'yellow bottle', 'red can', 'red cup', 'apple', 'barricade', 'VR glasses', 
#                                 'white bottle', 'black robot arm', 'plants'], points_dataloader, linguistic = model_type)
#for i, query in enumerate(['bowl', 'green bottle', 'ping pong ball', 'yellow bottle', 'red can', 'red cup', 'apple', 'barricade', 'VR glasses', 
#                'white bottle', 'black robot arm', 'plants']):
#    print(max_points[i], query)
