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
from omegaconf import OmegaConf

class CFLocalizer():
    def __init__(self, config_path, device = 'cuda', image_rep_size = 512, text_rep_size=768):
        self.device = device
        config = OmegaConf.load(config_path)
        self.model_type = config.web_models.segmentation
        self.dataset_path = os.path.join('clip-fields', config.saved_dataset_path)
        if self.model_type != 'owl':
            self.model_name = 'ViT-B/32'
        else:
            self.model_name = 'google/owlvit-base-patch32'
        self.sentence_model_name = 'all-mpnet-base-v2'
        self.clip_model, self.preprocessor, self.sentence_model = self.load_pretrained()
        self.points_dataloader = self.get_dataloader(self.dataset_path)
        max_coords, _ = self.points_dataloader.dataset.max(dim=0)
        min_coords, _ = self.points_dataloader.dataset.min(dim=0)
        model_weights_path = os.path.join('clip-fields', config.save_directory, 'implicit_scene_label_model_latest.pt')
        self.label_model = self.load_field(
            config,
            max_coords,
            min_coords,
            model_weights_path, 
            image_rep_size, 
            text_rep_size
        )

    def load_pretrained(self):
        if self.model_type == 'owl':
            model = OwlViTModel.from_pretrained(self.model_name).to(self.device)
            preprocessor = AutoProcessor.from_pretrained(self.model_name)
        else:
            model, preprocessor = clip.load(self.model_name, device=self.device)
        sentence_model = SentenceTransformer(self.sentence_model_name)
        return model, preprocessor, sentence_model

    def get_dataloader(self, cf_path):
        training_data = torch.load(cf_path)
        data_xyzs = training_data._label_xyz
        batch_size = 30_000
        points_dataloader = DataLoader(
            data_xyzs.detach().cpu(), batch_size=batch_size, num_workers=10,
        )
        print("Created data loader", points_dataloader)
        #merged_pcd = o3d.geometry.PointCloud()
        #merged_pcd.points = o3d.utility.Vector3dVector(points_dataloader.dataset)
        #merged_downpcd = merged_pcd.voxel_down_sample(voxel_size=0.03)
        #o3d.io.write_point_cloud(f"point_cloud.ply", merged_downpcd)
        return points_dataloader

    
    def load_field( 
        self,
        config,
        max_coords,
        min_coords,
        model_weights_path, 
        image_rep_size, 
        text_rep_size
    ):
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
        ).to(self.device)
        model_weights = torch.load(model_weights_path, map_location=self.device)
        model.load_state_dict(model_weights["model"])
        return model

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
            all_st_tokens = torch.from_numpy(self.sentence_model.encode(queries))
            all_st_tokens = F.normalize(all_st_tokens, p=2, dim=-1).to(self.device)
        return all_clip_tokens, all_st_tokens
        
    def find_alignment_over_model(self, queries, vision_weight = 10.0, text_weight = 1.0):
        clip_text_tokens, st_text_tokens = self.calculate_clip_and_st_embeddings_for_queries(queries)
        point_alignments = []
        with torch.no_grad():
            for data in tqdm.tqdm(self.points_dataloader, total = len(self.points_dataloader)):
                # Find alignmnents with the vectors
                predicted_label_latents, predicted_image_latents = self.label_model(data.to(self.device))
                data_text_tokens = F.normalize(predicted_label_latents, p=2, dim=-1).to(self.device)
                text_alignment = data_text_tokens @ st_text_tokens.T
                data_visual_tokens = F.normalize(predicted_image_latents, p=2, dim=-1).to(self.device)
                visual_alignment = data_visual_tokens @ clip_text_tokens.T
                total_alignment = (text_weight * text_alignment) + (vision_weight * visual_alignment)
                total_alignment /= (text_weight + vision_weight)
                point_alignments.append(total_alignment)

        point_alignments = torch.cat(point_alignments).T
        print(point_alignments.shape)
        return point_alignments

    # Currently we only support compute one query each time, in the future we might want to support check many queries

    def localize_AonB(self, A, B, k_A = 10, k_B = 50, data_type = 'r3d'):
        print("A is ", A)
        print("B is ", B)
        if B is None or B == '':
            target = self.find_alignment_for_A([A])[0]
        else:
            #points, _, _, _ = self.voxel_pcd.get_pointcloud()
            alignments = self.find_alignment_over_model([A, B]).cpu()
            A_points = self.points_dataloader.dataset[alignments[0].topk(k = k_A, dim = -1).indices].reshape(-1, 3)
            B_points = self.points_dataloader.dataset[alignments[1].topk(k = k_B, dim = -1).indices].reshape(-1, 3)
            distances = torch.norm(A_points.unsqueeze(1) - B_points.unsqueeze(0), dim=2)
            target = A_points[torch.argmin(torch.min(distances, dim = 1).values)]
        if data_type == 'r3d':
            target = target[[0, 2, 1]]
            target[1] = -target[1]
        return target

    def find_alignment_for_A(self, A):
        alignments = self.find_alignment_over_model(A).cpu()
        return self.points_dataloader.dataset[alignments.argmax(dim = -1)]