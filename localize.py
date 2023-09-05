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

from transformers import AutoProcessor, OwlViTModel

DEVICE = "cuda"

os.environ["TOKENIZERS_PARALLELISM"] = '(true | false)'

from transformers import AutoProcessor, OwlViTModel
from omegaconf import OmegaConf

import ml.api as ml
import usa
from usa.tasks.datasets.posed_rgbd import get_posed_rgbd_dataset, iter_xyz
from usa.models.point2emb import Point2EmbModel, Point2EmbModelConfig
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

model = None
preprocessor = None
sentence_model = None
lerf_model = None

def pseudo_lerf(lerf_model, points, scale = 0.5):
    #global lerf_model
    hashgrid = torch.concat([e(points) for e in lerf_model.clip_encs], dim = -1)
    return lerf_model.clip_net(torch.cat([hashgrid, torch.tensor([[scale]] * points.shape[0]).cuda()], dim = -1)).float()

# model type can be owl, openclip (only openclip requires weight_name), or clip
# if model_type is 'owl', then default model_name is 'google/owlvit-base-patch32'
# if model_type is 'openclip' or clip, then default model_name is 'ViT-B/32'
def load_pretrained(
    field_type = 'cf', 
    model_type = 'owl', 
    model_name = 'google/owlvit-base-patch32',
    weight_name = 'laion2b_s34b_b79k', 
    sentence_model_name = 'all-mpnet-base-v2'):
    global model, preprocessor, sentence_model, DEVICE
    if field_type == 'cf':
        sentence_model = SentenceTransformer(sentence_model_name)
    if model_type == 'owl':
        model = OwlViTModel.from_pretrained(model_name).to(DEVICE)
        preprocessor = AutoProcessor.from_pretrained(model_name)
    elif model_type == 'openclip':
        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained = weight_name,
        )
        preprocessor = open_clip.get_tokenizer('ViT-B-32')
    else:
        model, preprocess = clip.load(model_name, device=DEVICE)

# For this part, if field type is clip-fields or lerf, then, You need to provide your model_weights_path
# if your field type is usa-net, we will load your training config
# If you are using clip VIT-B/32, image rep size is 512
# For label rep, we are only considering SentenceTransformer("all-mpnet-base-v2"), so the label_rep_size is 768
# If you are loading lerf, then feel free to set config_path None
def load_field(
    field_type, 
    model_weights_path, 
    config_path = None, 
    image_rep_size = 512, 
    text_rep_size=768
):
    global DEVICE, lerf_model
    if config_path:
        config = OmegaConf.load(config_path)
    if field_type == 'cf':
        model = GridCLIPModel(
            image_rep_size = image_rep_size,
            text_rep_size = text_rep_size,
            mlp_depth = config.mlp_depth,
            mlp_width = config.mlp_width,
            log2_hashmap_size = config.log2_hashmap_size,
            num_levels = config.num_grid_levels,
            level_dim = config.level_dim,
            per_level_scale = config.per_level_scale,
        ).to(DEVICE)
        model_weights = torch.load(model_weights_path, map_location=DEVICE)
        model.load_state_dict(model_weights["model"])
        return model
    elif field_type == 'usa':
        config = Point2EmbModelConfig(**cfg.model)
        model = Point2EmbModel(config)
        model = model.to(DEVICE)
        model_weights = torch.load(model_weights_path, map_location=DEVICE)
        model.load_state_dict(model_weights["model"])
        return model
    else:
        config = LERFModelConfig()
        model = LERFField(config.hashgrid_layers, config.hashgrid_sizes, config.hashgrid_resolutions, 512)
        weights = torch.load(model_weights_path)
        field_weight = {}
        for weight in weights['pipeline']:
            if "lerf" in weight:
                field_weight[weight[len('_model.lerf_field.'):]] = weights['pipeline'][weight]
        model.load_state_dict(field_weight)
        lerf_model = model.to(DEVICE)
        return pseudo_lerf
    

def get_dataloader(r3d_path = None, cf_path = None):
    # For this part, you can either choose to load clip fields training data
    # or a r3d file to get an USA-Net styled data
    assert r3d_path or cf_path, 'you must provide a path to either your r3d file or loaded cf dataset'
    if cf_path:
        training_data = torch.load(cf_path)
        data_xyzs = training_data._label_xyz
    else:
        ds = get_posed_rgbd_dataset(key = 'r3d', path = r3d_path)
        data_xyzs = []
        for xyz, mask_tensor in iter_xyz(ds, 'data'):
            data = xyz[~mask_tensor]
            data = data[torch.randperm(len(data))[:int(len(data) * 0.1)]]
            data_xyzs.append(data)
        data_xyzs = torch.vstack(data_xyzs)
    batch_size = 300_000
    points_dataloader = DataLoader(
        data_xyzs.detach().cpu(), batch_size=batch_size, num_workers=10,
    )
    print("Created data loader", points_dataloader)
    return points_dataloader

def calculate_clip_and_st_embeddings_for_queries(queries, model_type = 'owl', st_embeddings = True):
    global model, preprocessor
    with torch.no_grad():
        if model_type == 'owl':
            inputs = preprocessor(
                text=[queries], return_tensors="pt"
            )
            inputs['input_ids'] = inputs['input_ids'].cuda()
            inputs['attention_mask'] = inputs['attention_mask'].cuda()
            all_clip_tokens = model.get_text_features(**inputs)
        elif model_type == 'openclip':
            all_clip_queries = preprocessor(queries)
            all_clip_tokens = model.encode_text(all_clip_queries).float().to(DEVICE)
        else:
            all_clip_queries = clip.tokenize(queries)
            all_clip_tokens = model.encode_text(all_clip_queries.to(DEVICE)).float()
        all_clip_tokens = F.normalize(all_clip_tokens, p=2, dim=-1)
        if st_embeddings:
            all_st_tokens = torch.from_numpy(sentence_model.encode(queries))
            all_st_tokens = F.normalize(all_st_tokens, p=2, dim=-1).to(DEVICE)
        else:
            all_st_tokens = None
    return all_clip_tokens, all_st_tokens

def find_alignment_over_model(label_model, queries, dataloader, model_type, 
            vision_weight = 1.0, text_weight = 10.0, linguistic = 'owl', relevancy = False,
            lerf_scale = 0.5):
    global lerf_model
    clip_text_tokens, st_text_tokens = calculate_clip_and_st_embeddings_for_queries(
        queries, linguistic, True if model_type == 'cf' else False)
    if relevancy:
        negtives = ['objects', 'things', 'stuff', 'texture']
        negative_clip_text_tokens, negative_st_text_tokens = calculate_clip_and_st_embeddings_for_queries(
            negtives, linguistic, True if model_type == 'cf' else False)
    # We give different weights to visual and semantic alignment 
    # for different types of queries.
    point_opacity = []
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader, total = len(dataloader)):
            # Find alignmnents with the vectors
            if model_type == 'cf':
                predicted_label_latents, predicted_image_latents = label_model(data.to(DEVICE))
            elif model_type == 'usa':
                predicted_image_latents = label_model(data.to(DEVICE))[:, :-1]
            else:
                predicted_image_latents = label_model(lerf_model, data.to(DEVICE), scale = lerf_scale)
                del data
            if model_type == 'cf':
                data_text_tokens = F.normalize(predicted_label_latents, p=2, dim=-1).to(DEVICE)
                text_alignment = data_text_tokens @ st_text_tokens.T
            data_visual_tokens = F.normalize(predicted_image_latents, p=2, dim=-1).to(DEVICE)
            visual_alignment = data_visual_tokens @ clip_text_tokens.T
            if relevancy:
                if model_type == 'cf':
                    negative_text_alignment = data_text_tokens @ negative_st_text_tokens.T
                negative_visual_alignment = data_visual_tokens @ negative_clip_text_tokens.T
                bs, qs = visual_alignment.shape
                _, ns = negative_visual_alignment.shape
                if model_type == 'cf':
                    text_relevancy_scores = (torch.exp(text_alignment.unsqueeze(-1).expand(bs, qs, ns)) / 
                        (torch.exp(text_alignment.unsqueeze(-1).expand(bs, qs, ns)) + 
                        torch.exp(negative_text_alignment.unsqueeze(-2).expand(bs, qs, ns)))).min(-1).values
                visual_relevancy_scores = (torch.exp(visual_alignment.unsqueeze(-1).expand(bs, qs, ns)) / 
                    (torch.exp(visual_alignment.unsqueeze(-1).expand(bs, qs, ns)) + 
                     torch.exp(negative_visual_alignment.unsqueeze(-2).expand(bs, qs, ns)))).min(-1).values
                if model_type == 'cf':
                    total_alignment = (text_weight * text_relevancy_scores) + vision_weight * visual_relevancy_scores
                    total_alignment /= (text_weight + vision_weight)
                else: 
                    total_alignment = visual_relevancy_scores
            else:
                if model_type == 'cf':
                    total_alignment = (text_weight * text_alignment) + (vision_weight * visual_alignment)
                    total_alignment /= (text_weight + vision_weight)
                else:
                    total_alignment = visual_alignment
            point_opacity.append(total_alignment)

    point_opacity = torch.cat(point_opacity).T
    print(point_opacity.shape)
    return point_opacity

FIELD_TYPE = 'lerf'
CLIP_TYPE = 'ViT-B/32'
OWL_TYPE = 'google/owlvit-base-patch32'
CF_PATH = 'clip-fields/kitchen_owl1/implicit_scene_label_model_latest.pt'
USA_WEIGHT_PATH = 'usa/usa/4_256_no/run_0/checkpoints/ckpt.8000.pt'
LERF_WEIGHT_PATH = 'lerf/outputs/Kitchen/lerf/2023-09-04_155930/nerfstudio_models/step-000029999.ckpt'
load_pretrained(field_type = FIELD_TYPE, model_type = 'openclip', model_name = 'ViT-B/32')
label_model = load_field(field_type = FIELD_TYPE, config_path = None, model_weights_path = LERF_WEIGHT_PATH)
points_dataloader = get_dataloader(r3d_path = 'clip-fields/Kitchen.r3d')
for lerf_scale in np.arange(0.1, 2.0, 0.3):
    find_alignment_over_model(label_model, ['Table', 'Chair'], points_dataloader,
        linguistic = 'openclip', model_type = FIELD_TYPE, lerf_scale = 0.5, relevancy = True)
#find_alignment_over_model(label_model, ['Table', 'Chair'], points_dataloader, model_type = 'usa')