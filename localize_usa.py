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

class USANetLocalizer():
    def __init__(self, config_path, device = 'cuda'):
        self.device = device
        config = OmegaConf.load(config_path)
        weight_path = os.path.join('usa', 
            config.trainer.base_run_dir, 
            config.trainer.exp_name,
            'run_' + str(config.trainer.run_id),
            "checkpoints",
            "ckpt.pt")
        print(weight_path)
        self.dataset_path = os.path.join('usa', config.task.dataset_path)
        # for now we do not consider owl-vit in USA-Net, yet we strongly believe owl-vit is promising for USA-Net's future
        self.model_type = "clip"
        if self.model_type != 'owl':
            self.model_name = 'ViT-B/32'
        else:
            self.model_name = 'google/owlvit-base-patch32'
        self.clip_model, self.preprocessor = self.load_pretrained()
        self.points_dataloader = self.get_dataloader(self.dataset_path)
        self.label_model = self.load_field(config_path = config_path, model_weights_path = weight_path)

    def load_pretrained(self):
        if self.model_type == 'owl':
            model = OwlViTModel.from_pretrained(self.model_name).to(self.device)
            preprocessor = AutoProcessor.from_pretrained(self.model_name)
        else:
            model, preprocessor = clip.load(self.model_name, device=self.device)
        return model, preprocessor

    def get_dataloader(self, r3d_path, sample_prob = 0.01):
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
        return points_dataloader

    
    def load_field( 
        self,
        model_weights_path, 
        config_path, 
    ):
        config = OmegaConf.load(config_path)
        config = Point2EmbModelConfig(**config.model)
        model = Point2EmbModel(config)
        model = model.to(self.device)
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
        return all_clip_tokens
        
    def find_alignment_over_model(self, queries):
        clip_text_tokens = self.calculate_clip_and_st_embeddings_for_queries(queries)
        point_alignments = []
        with torch.no_grad():
            for data in tqdm.tqdm(self.points_dataloader, total = len(self.points_dataloader)):
                # Find alignmnents with the vectors
                predicted_image_latents = self.label_model(data.to(self.device))[:, :-1]
                data_visual_tokens = F.normalize(predicted_image_latents, p=2, dim=-1).to(self.device)
                visual_alignment = data_visual_tokens @ clip_text_tokens.T
                point_alignments.append(visual_alignment)

        point_alignments = torch.cat(point_alignments).T
        print(point_alignments.shape)
        return point_alignments

    # Currently we only support compute one query each time, in the future we might want to support check many queries

    def localize_AonB(self, A, B, k_A = 10, k_B = 50):
        print("A is ", A)
        print("B is ", B)
        if B is None or B == '':
            return self.find_alignment_for_A([A])[0]
        else:
            #points, _, _, _ = self.voxel_pcd.get_pointcloud()
            alignments = self.find_alignment_over_model([A, B]).cpu()
            A_points = self.points_dataloader.dataset[alignments[0].topk(k = k_A, dim = -1).indices].reshape(-1, 3)
            B_points = self.points_dataloader.dataset[alignments[1].topk(k = k_B, dim = -1).indices].reshape(-1, 3)
            distances = torch.norm(A_points.unsqueeze(1) - B_points.unsqueeze(0), dim=2)
            target = A_points[torch.argmin(torch.min(distances, dim = 1).values)]
            return target

    def find_alignment_for_A(self, A):
        alignments = self.find_alignment_over_model(A).cpu()
        return self.points_dataloader.dataset[alignments.argmax(dim = -1)]