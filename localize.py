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
lerf_model = None
bounds = None

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
    text_rep_size=768,
    max_coords = None,
    min_coords = None
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
            max_coords = max_coords,
            min_coords = min_coords
        ).to(DEVICE)
        model_weights = torch.load(model_weights_path, map_location=DEVICE)
        model.load_state_dict(model_weights["model"])
        return model
    elif field_type == 'usa':
        config = Point2EmbModelConfig(**config.model)
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
    # bounds is needed to test LERF, so if you choose to load dataloader from cf_path, you should not
    # be able to test LERF
    global bounds
    assert r3d_path or cf_path, 'you must provide a path to either your r3d file or loaded cf dataset'
    if cf_path:
        training_data = torch.load(cf_path)
        data_xyzs = training_data._label_xyz
    else:
        ds = get_posed_rgbd_dataset(key = 'r3d', path = r3d_path)
        bounds = get_bounds(ds)
        #map = get_ground_truth_map_from_dataset(ds, 0.05, (-1, 0))
        #points = []
        #for i in range(map.grid.shape[0]):
        #    for j in range(map.grid.shape[1]):
        #        if map.grid[i, j]:
        #            points.append(torch.tensor(map.to_xy((i, j))))
        #z_values = torch.arange(bounds.zmin + 0.5, bounds.zmax - 0.5, 0.2).unsqueeze(1)
        #data_xyzs = torch.cat((torch.stack(points).repeat(len(z_values), 1), z_values.repeat(len(points), 1)), dim=1)
        data_xyzs = []
        for xyz, mask_tensor in iter_xyz(ds, 'data'):
            data = xyz[~mask_tensor]
            data = data[torch.randperm(len(data))[:int(len(data) * 0.01)]]
            data_xyzs.append(data)
        data_xyzs = torch.vstack(data_xyzs)
    batch_size = 30_000
    points_dataloader = DataLoader(
        data_xyzs.detach().cpu(), batch_size=batch_size
    )
    print("Created data loader", points_dataloader)
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(points_dataloader.dataset)
    merged_downpcd = merged_pcd.voxel_down_sample(voxel_size=0.03)
    o3d.io.write_point_cloud(f"nyu_lab.ply", merged_downpcd)
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
            # for clip-fields
            vision_weight = 10.0, text_weight = 10.0, linguistic = 'owl', 
            # for lerf, but can also be for USA and CF
            relevancy = False,
            # for lerf specifically
            lerf_scale = 0.5):
    global lerf_model, DEVICE, bounds
    clip_text_tokens, st_text_tokens = calculate_clip_and_st_embeddings_for_queries(
        queries, linguistic, True if model_type == 'cf' else False)
    if relevancy:
        negtives = ['objects', 'things', 'stuff', 'texture']
        negative_clip_text_tokens, negative_st_text_tokens = calculate_clip_and_st_embeddings_for_queries(
            negtives, linguistic, True if model_type == 'cf' else False)
    # We give different weights to visual and semantic alignment 
    # for different types of queries.
    point_alignments = []
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader, total = len(dataloader)):
            # Find alignmnents with the vectors
            if model_type == 'cf':
                # We assume clip-fields models will be run on clip-fields dataset
                # If you run it on USA-Net dataset, uncomment these two lines of codes
                #data = data[:, [0, 2, 1]]
                #data[:, 2] = -data[:, 2]
                predicted_label_latents, predicted_image_latents = label_model(data.to(DEVICE))
            elif model_type == 'usa':
                predicted_image_latents = label_model(data.to(DEVICE))[:, :-1]
            else:
                #data = data[:, [0, 2, 1]]
                #data[:, 1] = -data[:, 1]
                data[:, 0] = (data[:, 0] - bounds.xmin) / (bounds.xmax - bounds.xmin) 
                data[:, 1] = (data[:, 1] - bounds.ymin) / (bounds.ymax - bounds.ymin) 
                data[:, 2] = (data[:, 2] - bounds.zmin) / (bounds.zmax - bounds.zmin) 
                predicted_image_latents = label_model(lerf_model, data.to(DEVICE), scale = lerf_scale)
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
            point_alignments.append(total_alignment)

    point_alignments = torch.cat(point_alignments).T
    print(point_alignments.shape)
    return point_alignments

# Currently we only support compute one query each time, in the future we might want to support check many queries

# Keep in mind that the objective is picking up A from B
def find_alignment_for_B(label_model, B, dataloader, model_type, 
            # for this function only, how many points do you want to considered as relevant to B
            threshold_precentile = 0.05,
            # for clip-fields
            vision_weight = 10.0, text_weight = 10.0, linguistic = 'owl', 
            # for lerf, but can also be for USA and CF
            relevancy = True,
            # for lerf specifically
            lerf_scale = 0.5):
    assert threshold_precentile > 0 and threshold_precentile <= 1, 'invalid threshold_precentile'
    alignments = find_alignment_over_model(label_model, B, dataloader, model_type,
                vision_weight, text_weight, linguistic,
                relevancy,
                lerf_scale).cpu()
    threshold = int(len(dataloader.dataset) * threshold_precentile)
    B_dataset = dataloader.dataset[alignments.topk(k = threshold, dim = -1).indices]
    BATCH_SIZE = 3000
    B_dataset = B_dataset.reshape(-1, 3)
    return DataLoader(B_dataset, batch_size = BATCH_SIZE, num_workers = 10)

def lerf_scales_processing(lerf_scales):
    compute_scales = list(set(lerf_scales))
    scale_to_inds = {k: v for v, k in enumerate(compute_scales)}
    lerf_scale_inds = [scale_to_inds[lerf_scale] for lerf_scale in lerf_scales]
    return compute_scales, lerf_scale_inds

def find_alignment_for_A(label_model, A, dataloader, model_type, 
            # for clip-fields
            vision_weight = 10.0, text_weight = 10.0, linguistic = 'owl', 
            # for lerf, but can also be for USA and CF
            relevancy = False,
            # for lerf specifically
            lerf_scale = 0.5):
    if type(lerf_scale) == type(0.0) or model_type != 'lerf':
        alignments = find_alignment_over_model(label_model, A, dataloader, model_type,
                vision_weight, text_weight, linguistic,
                relevancy,
                lerf_scale).cpu()
        return dataloader.dataset[alignments.argmax(dim = -1)]
    else:
        compute_scales, lerf_scale_inds = lerf_scales_processing(lerf_scale)
        alignments = []
        #print(lerf_scale, lerf_scale_inds, compute_scales)
        for scale in compute_scales:
            alignments.append(find_alignment_over_model(label_model, A, dataloader, model_type,
                vision_weight, text_weight, linguistic,
                relevancy,
                scale).cpu())
        alignments = torch.stack(alignments)
        final_alignments = []
        for i in range(len(lerf_scale_inds)):
            #print(lerf_scale_inds[i], lerf_scale[i], i, alignments.shape)
            final_alignments.append(alignments[lerf_scale_inds[i], i, :])
        final_alignments = torch.stack(final_alignments, dim = 0)
        return dataloader.dataset[final_alignments.argmax(dim = -1)]
        

# Note that even though we cannot localize many queries with LERF at the same time,
# we can actually select lerf scales for many queries at the same time.
def scale_selection(label_model, queries, dataloader,
        linguistic = 'openclip', model_type = 'lerf', relevancy = True):
    relevancy_scores = []
    for lerf_scale in np.arange(0.1, 2.0, 0.3):
        relevancy_scores.append(
            find_alignment_over_model(label_model, queries, dataloader,
            linguistic = linguistic, model_type = model_type, lerf_scale = lerf_scale, 
            relevancy = relevancy).max(dim = -1).values
        )
    scales = torch.arange(0.1, 2.0, 0.3).to(DEVICE)[torch.stack(relevancy_scores, dim = 1).squeeze(-1).argmax(dim = -1)]    
    return scales

FIELD_TYPE = 'usa'
MODEL_TYPE = 'clip'
DATASET_PATH = 'clip-fields/detic_labeled_dataset.pt'
if MODEL_TYPE != 'owl':
    MODEL_NAME = 'ViT-B/32'
else:
    MODEL_NAME = 'google/owlvit-base-patch32'
if FIELD_TYPE == 'cf':
    WEIGHT_PATH = 'clip-fields/kitchen_owl/implicit_scene_label_model_latest.pt'
    CONFIG_PATH = 'clip-fields/configs/train.yaml'
if FIELD_TYPE == 'usa':
    WEIGHT_PATH = 'usa/usa/4_256_no/run_0/checkpoints/ckpt.8000.pt'
    CONFIG_PATH = 'usa/configs/train.yaml'
if FIELD_TYPE == 'lerf':
    WEIGHT_PATH = 'lerf/outputs/Kitchen/lerf/2023-09-04_155930/nerfstudio_models/step-000029999.ckpt'
    CONFIG_PATH = None
load_pretrained(field_type = FIELD_TYPE, model_type = MODEL_TYPE , model_name = MODEL_NAME)

points_dataloader = get_dataloader(r3d_path = 'clip-fields/Kitchen.r3d')
#points_dataloader = get_dataloader(cf_path = DATASET_PATH)

max_coords, _ = points_dataloader.dataset.max(dim=0)
min_coords, _ = points_dataloader.dataset.min(dim=0)
#max_coords = torch.tensor([bounds.xmax, bounds.zmax, -bounds.ymin])
#min_coords = torch.tensor([bounds.xmin, bounds.zmin, -bounds.ymax])

label_model = load_field(field_type = FIELD_TYPE, config_path = CONFIG_PATH, model_weights_path = WEIGHT_PATH, max_coords = max_coords, min_coords = min_coords)
#scales = scale_selection(label_model, 'Chair', points_dataloader,
#        linguistic = MODEL_TYPE, model_type = FIELD_TYPE, relevancy = True)

eval_data = pd.read_csv('clip-fields/kitchen.csv')
queries = list(eval_data['query'])

xs, ys, zs, affords = list(eval_data['x']), list(eval_data['y']), list(eval_data['z']), list(eval_data['affordance'])

xyzs = torch.stack([torch.tensor(xs), torch.tensor(zs), -torch.tensor(ys)], dim = 1)
#xyzs = torch.stack([torch.tensor(xs), torch.tensor(ys), torch.tensor(zs)], dim = 1)

if FIELD_TYPE == 'lerf':
    max_points = find_alignment_for_A(label_model, queries, points_dataloader, FIELD_TYPE, 
            vision_weight = 1.0, text_weight = 10.0, linguistic = MODEL_TYPE, relevancy = False,
            lerf_scale = scale_selection(label_model, queries, points_dataloader, relevancy = False).detach().cpu().tolist())
else:
    max_points = find_alignment_for_A(label_model, queries, points_dataloader, FIELD_TYPE, 
            vision_weight = 10.0, text_weight = 10.0, linguistic = MODEL_TYPE, relevancy = False)
print(max_points.shape)
for max_point, query in zip(max_points, queries):
    print(max_point, query)

correctness = torch.linalg.norm((max_points[:, [0, 2]] - xyzs[:, [0, 2]]), dim = -1) <= torch.tensor(affords)
print(np.array(queries)[torch.where(correctness)[0].numpy()], 
    np.array(queries)[torch.where(~correctness)[0].numpy()], 
    len(np.array(queries)[torch.where(correctness)[0].numpy()]) / len(correctness))


#dataloader = find_alignment_for_B(label_model, ['Chair'], points_dataloader, FIELD_TYPE, 
            # for this function only, how many points do you want to considered as relevant to B
#            threshold_precentile = 0.01,
            # for clip-fields
#            vision_weight = 10.0, text_weight = 10.0, linguistic = MODEL_TYPE, 
            # for lerf, but can also be for USA and CF
#            relevancy = True,
            # for lerf specifically
#            lerf_scale = scales.item())

#print(find_alignment_for_A(label_model, ['mustard (yellow bottle)'], points_dataloader, FIELD_TYPE, 
#            # for clip-fields
#            vision_weight = 10.0, text_weight = 10.0, linguistic = MODEL_TYPE, 
#            # for lerf, but can also be for USA and CF
#            relevancy = False,
#            # for lerf specifically
#            lerf_scale = scales.item()))

#from matplotlib import pyplot as plt
#fig, ax = plt.subplots(1, 1)
#ax.scatter(dataloader.dataset[:, 0], dataloader.dataset[:, 1])
#fig.savefig('foo.jpg')
#relevancy_scores = []
#for lerf_scale in np.arange(0.1, 2.0, 0.8):
#    relevancy_scores.append(find_alignment_over_model(label_model, 'Table', points_dataloader,
#        linguistic = 'openclip', model_type = FIELD_TYPE, lerf_scale = lerf_scale, relevancy = True).max(dim = -1).values)
#scales = torch.arange(0.1, 2.0, 0.3).to(DEVICE)[torch.stack(relevancy_scores, dim = 1).squeeze(-1).argmax(dim = -1)]
#print(scales.item())
#find_alignment_over_model(label_model, ['Table', 'Chair'], points_dataloader, model_type = FIELD_TYPE)
