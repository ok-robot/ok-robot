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

from home_robot.utils.voxel import VoxelizedPointcloud

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
    #sentence_model_name = 'all-mpnet-base-v2',
    device = 'cuda'):
    global model, preprocessor, sentence_model
    #sentence_model = SentenceTransformer(sentence_model_name)
    if model_type == 'owl':
        model = OwlViTModel.from_pretrained(model_name).to(device)
        preprocessor = AutoProcessor.from_pretrained(model_name)
    else:
        model, preprocessor = clip.load(model_name, device=device)
    #return model, preprocessor, sentence_model
    return model, preprocessor

# For this part, if field type is clip-fields or lerf, then, You need to provide your model_weights_path
# if your field type is usa-net, we will load your training config
# If you are using clip VIT-B/32, image rep size is 512
# For label rep, we are only considering SentenceTransformer("all-mpnet-base-v2"), so the label_rep_size is 768
# If you are loading lerf, then feel free to set config_path None
def load_pcd(cf_path):
    training_data = torch.load(cf_path)
    voxel_pcd = VoxelizedPointcloud()
    voxel_pcd.add(points = training_data._label_xyz, 
              features = training_data._image_features,
              rgb = training_data._label_rgb,
              weights = training_data._label_weight)
    return voxel_pcd
    

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

def calculate_clip_and_st_embeddings_for_queries(queries, model, preprocessor, device = 'cuda', model_type = 'owl'):
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
        #all_st_tokens = torch.from_numpy(sentence_model.encode(queries))
        #all_st_tokens = F.normalize(all_st_tokens, p=2, dim=-1).to(device)
    #return all_clip_tokens, all_st_tokens
    return all_clip_tokens

#def find_alignment_over_model(voxel_pcd, clip_model, preprocessor, sentence_model, queries, dataloader,
#                 linguistic = 'owl', device = 'cuda'):
def find_alignment_over_model(voxel_pcd, clip_model, preprocessor, queries, linguistic = 'owl', device = 'cuda'):
    clip_text_tokens = calculate_clip_and_st_embeddings_for_queries(
            queries, clip_model, preprocessor, model_type = linguistic, device = device)
    points, features, _, _ = voxel_pcd.get_pointcloud()
    features = F.normalize(features, p=2, dim=-1)
    point_alignments = clip_text_tokens.detach().cpu() @ features.T
    
    print(point_alignments.shape)
    return point_alignments

# Currently we only support compute one query each time, in the future we might want to support check many queries

def localize_AonB(voxel_pcd, clip_model, preprocessor, A, B, k_A = 10, k_B = 50,
        linguistic = 'clip', data_type = 'r3d', device = 'cuda'):
    print("A is ", A)
    print("B is ", B)
    if B is None or B == '':
        target = find_alignment_for_A(voxel_pcd, clip_model, preprocessor, [A], linguistic = linguistic)[0]
    else:
        points, _, _, _ = voxel_pcd.get_pointcloud()
        alignments = find_alignment_over_model(voxel_pcd, clip_model, preprocessor, [A, B], linguistic = linguistic).cpu()
        A_points = points[alignments[0].topk(k = k_A, dim = -1).indices].reshape(-1, 3)
        B_points = points[alignments[1].topk(k = k_B, dim = -1).indices].reshape(-1, 3)
        distances = torch.norm(A_points.unsqueeze(1) - B_points.unsqueeze(0), dim=2)
        target = A_points[torch.argmin(torch.min(distances, dim = 1).values)]
    if data_type == 'r3d':
        target = target[[0, 2, 1]]
        target[1] = -target[1]
    return target

def find_alignment_for_A(voxel_pcd, clip_model, preprocessor, A, linguistic = 'owl'):
    points, features, _, _ = voxel_pcd.get_pointcloud()
    alignments = find_alignment_over_model(voxel_pcd, clip_model, preprocessor,
                A, linguistic).cpu()
    return points[alignments.argmax(dim = -1)]
        

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
    clip_model, preprocessor = load_pretrained(model_type = model_type , model_name = model_name)
    voxel_pcd = load_pcd(cf_path = dataset_path)
    return voxel_pcd, clip_model, preprocessor, model_type

#label_model, clip_model, preprocessor, sentence_model, points_dataloader, model_type = load_everything('clip-fields/configs/train.yaml')
#print(localize_AonB(label_model, clip_model, preprocessor, sentence_model, 'plant', '', points_dataloader, k_A = 10, k_B = 1000, linguistic = model_type))
#evaluate()
