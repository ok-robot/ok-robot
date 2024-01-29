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

import pandas as pd
import clip
import pandas as pd

from transformers import AutoProcessor, OwlViTModel
import sys
sys.path.append('voxel-map')

DEVICE = "cuda"

os.environ["TOKENIZERS_PARALLELISM"] = '(true | false)'
from omegaconf import OmegaConf

from home_robot.utils.voxel import VoxelizedPointcloud

class VoxelMapLocalizer():
    def __init__(self, config_path, device = 'cuda'):
        self.device = device
        config = OmegaConf.load(config_path)
        self.model_type = config.web_models.segmentation
        self.dataset_path = os.path.join('voxel-map', config.cache_path)
        if self.model_type != 'owl':
            self.model_name = 'ViT-B/32'
        else:
            self.model_name = 'google/owlvit-base-patch32'
        self.clip_model, self.preprocessor = self.load_pretrained()
        self.voxel_pcd = self.load_pcd()

    def load_pretrained(self):
        if self.model_type == 'owl':
            model = OwlViTModel.from_pretrained(self.model_name).to(self.device)
            preprocessor = AutoProcessor.from_pretrained(self.model_name)
        else:
            model, preprocessor = clip.load(self.model_name, device=self.device)
        return model, preprocessor

    
    def load_pcd(self):
        training_data = torch.load(self.dataset_path)
        voxel_pcd = VoxelizedPointcloud()
        voxel_pcd.add(points = training_data._label_xyz, 
              features = training_data._image_features,
              rgb = training_data._label_rgb,
              weights = training_data._label_weight)
        return voxel_pcd

    def calculate_clip_and_st_embeddings_for_queries(self, queries):
        with torch.no_grad():
            if self.model_type == 'owl':
                inputs = self.preprocessor(
                    text=[queries], return_tensors="pt"
                )
                inputs['input_ids'] = inputs['input_ids'].to(self.device)
                inputs['attention_mask'] = inputs['attention_mask'].to(self.device)
                all_clip_tokens = self.clip_model.get_text_features(**inputs)
            else:
                all_clip_queries = clip.tokenize(queries)
                all_clip_tokens = self.clip_model.encode_text(all_clip_queries.to(self.device)).float()
            all_clip_tokens = F.normalize(all_clip_tokens, p=2, dim=-1)
        return all_clip_tokens
        
    def find_alignment_over_model(self, queries):
        clip_text_tokens = self.calculate_clip_and_st_embeddings_for_queries(queries)
        points, features, _, _ = self.voxel_pcd.get_pointcloud()
        features = F.normalize(features, p=2, dim=-1)
        point_alignments = clip_text_tokens.detach().cpu() @ features.T
    
        print(point_alignments.shape)
        return point_alignments

    # Currently we only support compute one query each time, in the future we might want to support check many queries

    def localize_AonB(self, A, B, k_A = 10, k_B = 50,data_type = 'r3d'):
        print("A is ", A)
        print("B is ", B)
        if B is None or B == '':
            target = self.find_alignment_for_A([A])[0]
        else:
            points, _, _, _ = self.voxel_pcd.get_pointcloud()
            alignments = self.find_alignment_over_model([A, B]).cpu()
            A_points = points[alignments[0].topk(k = k_A, dim = -1).indices].reshape(-1, 3)
            B_points = points[alignments[1].topk(k = k_B, dim = -1).indices].reshape(-1, 3)
            distances = torch.norm(A_points.unsqueeze(1) - B_points.unsqueeze(0), dim=2)
            target = A_points[torch.argmin(torch.min(distances, dim = 1).values)]
        if data_type == 'r3d':
            target = target[[0, 2, 1]]
            target[1] = -target[1]
        return target

    def find_alignment_for_A(self, A):
        points, features, _, _ = self.voxel_pcd.get_pointcloud()
        alignments = self.find_alignment_over_model(A).cpu()
        return points[alignments.argmax(dim = -1)]
