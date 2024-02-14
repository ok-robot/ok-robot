'''
This file adapts from DeticDenseLabelledDataset 
    in Clip-fields (https://github.com/notmahi/clip-fields) project
 Most codes are adapted from:
    1. https://github.com/notmahi/clip-fields/blob/main/dataloaders/real_dataset.py
License:
MIT License

Copyright (c) 2024 Nur Muhammad "Mahi" Shafiullah

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import logging
from typing import List, Optional, Union
import clip
import einops
import os
import torch
import tqdm
from matplotlib import pyplot as plt
import cv2
import wget

import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from dataloaders.record3d import R3DSemanticDataset
from dataloaders.scannet_200_classes import CLASS_LABELS_200

from torch.utils.data import Dataset

# import some common libraries
import sys

import torchvision.transforms as transforms
from transformers import AutoProcessor, OwlViTForObjectDetection
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor



# New visualizer class to disable jitter.
import matplotlib.colors as mplc


def center_to_corners_format(bboxes_center):
    center_x, center_y, width, height = bboxes_center.unbind(-1)
    bbox_corners = torch.stack(
        # top left x, top left y, bottom right x, bottom right y
        [(center_x - 0.5 * width), (center_y - 0.5 * height), (center_x + 0.5 * width), (center_y + 0.5 * height)],
        dim=-1,
    )
    return bbox_corners

def post_process_object_detection(
    outputs, threshold: float = 0.1, target_sizes = None
):
    logits, boxes, class_embeddings = outputs.logits, outputs.pred_boxes, outputs.class_embeds
    
    if target_sizes is not None:
        if len(logits) != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )

    probs = torch.max(logits, dim=-1)
    scores = torch.sigmoid(probs.values)
    labels = probs.indices
    
    # Convert to [x0, y0, x1, y1] format
    boxes = center_to_corners_format(boxes)
    
    # Convert from relative [0, 1] to absolute [0, height] coordinates
    if target_sizes is not None:
        img_h, img_w = target_sizes.unbind(1)
        
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

    results = []
    for s, l, b, c in zip(scores, labels, boxes, class_embeddings):
        score = s[s > threshold]
        label = l[s > threshold]
        box = b[s > threshold]
        class_embed = c[s > threshold]
        results.append({"scores": score, "labels": label, "boxes": box, "class_embed": class_embed})
        
    return results

class OWLViTLabelledDataset(Dataset):
    #LSEG_LABEL_WEIGHT = 0.1
    #LSEG_IMAGE_DISTANCE = 10.0

    def __init__(
        self,
        view_dataset,
        owl_model_name: str = "google/owlvit-base-patch32",
        sam_model_type = "vit_b",
        device: str = "cuda",
        threshold: float = 0.1,
        subsample_prob: float = 0.2,
        visualize_results: bool = False,
        visualization_path: Optional[str] = None,
    ):
        dataset = view_dataset
        view_data = (
            view_dataset.dataset if isinstance(view_dataset, Subset) else view_dataset
        )
        self._image_width, self._image_height = view_data.image_size
        self._model = OwlViTForObjectDetection.from_pretrained(owl_model_name).to(device)
        self._processor = AutoProcessor.from_pretrained(owl_model_name)

        self._device = device
        self._owl_threshold = threshold
        self._subsample_prob = subsample_prob

        self._label_xyz = []
        self._label_rgb = []
        self._label_weight = []
        self._image_features = []
        self._distance = []

        self._visualize = visualize_results
        if self._visualize:
            assert visualization_path is not None
            self._visualization_path = Path(visualization_path)
            os.makedirs(self._visualization_path, exist_ok=True)
        # First, setup owl-vit with the combined classes.
        self._setup_owl_all_classes(view_data)
        # Next Load SAM models according to model config as owl-vit only provides bounding boxes and we need
        # SAM to obtain segmentation mask
        if sam_model_type == 'vit_b':
            url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
            sam_model_path_name = 'sam_vit_b_01ec64.pth'
        elif sam_model_type == 'vit_l':
            url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth'
            sam_model_path_name = 'sam_vit_l_0b3195.pth'
        else:
            sam_model_type = 'vit_h'
            url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
            sam_model_path_name = 'sam_vit_h_4b8939.pth'
        if not os.path.exists(sam_model_path_name):
            wget.download(url, out = sam_model_path_name)
        sam = sam_model_registry[sam_model_type](checkpoint=sam_model_path_name)
        mask_predictor = SamPredictor(sam)
        mask_predictor.model = mask_predictor.model.to(self._device)
        self._setup_owl_dense_labels(
            dataset, mask_predictor
        )

        del mask_predictor

    @torch.no_grad()
    def _setup_owl_dense_labels(
        self, dataset, mask_predictor
    ):
        # Now just iterate over the images and do Detic preprocessing.
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)
        label_idx = 0
        text_strings = [
            OWLViTLabelledDataset.process_text(x) for x in self._all_classes
        ]
        for idx, data_dict in tqdm.tqdm(
            enumerate(dataloader), total=len(dataset), desc="Calculating OWL features"
        ):
            rgb = einops.rearrange(data_dict["rgb"][..., :3], "b h w c -> b c h w")
            xyz = data_dict["xyz_position"]
            for image, coordinates in zip(rgb, xyz):
                # Now calculate the OWL-ViT detection for this.
                target_sizes = torch.Tensor([image[0].size()])
                inputs = self._processor(text=self._all_classes, images=image, return_tensors="pt")
                for input in inputs:
                    inputs[input] = inputs[input].to(self._device)
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    results = post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=self._owl_threshold)
                    i = 0
                    text = self._all_classes[i]
                    boxes, scores, labels, features = results[i]["boxes"], results[i]["scores"], results[i]["labels"], results[i]['class_embed']
                
                # Now run SAM to compute segmentation mask
                input_boxes = boxes.detach().to(mask_predictor.device)
                mask_predictor.set_image(image.permute(1, 2, 0).numpy())
                if len(input_boxes) == 0:
                    break
                transformed_boxes = mask_predictor.transform.apply_boxes_torch(input_boxes.reshape(-1, 4), image.shape[1:])  
                masks, iou_predictions, low_res_masks = mask_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False
                )
                #print(masks.shape)
                masks = masks[:, 0, :, :]

                reshaped_rgb = einops.rearrange(image, "c h w -> h w c")
                (
                    reshaped_coordinates,
                    valid_mask,
                ) = self._reshape_coordinates_and_get_valid(coordinates, data_dict)
                
                if self._visualize:
                    image_vis = np.array(image.permute(1, 2, 0))
                    image = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(self._visualization_path / f"{idx}_Clean.jpg"), image)
                    for score, box, label in zip(scores, boxes, labels):
                        tl_x, tl_y, br_x, br_y = box
                        tl_x, tl_y, br_x, br_y = tl_x.item(), tl_y.item(), br_x.item(), br_y.item()
                        cv2.rectangle(image_vis, (int(tl_x), int(tl_y)), (int(br_x), int(br_y)), (255, 0, 0), 2)
                        cv2.rectangle(image_vis, (int(tl_x), int(br_y)), (int(tl_x) + 200, int(br_y) + 13), (255, 255, 255), -1)
                        cv2.putText(
                            image_vis, f'{text_strings[label.item()]}: {score:1.2f}', (int(tl_x), int(br_y) + 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 2)
                    image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)    
                    segmentation_color_map = np.zeros(image_vis.shape, dtype=np.uint8)
                    #print(segmentation_color_map.shape, masks.shape, image_vis.shape)
                    for mask in masks:
                        segmentation_color_map[mask.detach().cpu().numpy()] = [0, 255, 0]
                    image_vis = cv2.addWeighted(image_vis, 0.7, segmentation_color_map, 0.3, 0)
                    cv2.imwrite(str(self._visualization_path / f"{idx}.jpg"), image_vis)

                for pred_class, pred_box, pred_score, feature, pred_mask in zip(
                    labels.cpu(),
                    boxes.cpu(),
                    scores.cpu(),
                    features.cpu(),
                    masks.cpu(),
                ):
                    img_h, img_w = target_sizes.unbind(1)
                    real_mask = pred_mask[valid_mask]
                    real_mask_rect = valid_mask & pred_mask
                    # Go over each instance and add it to the DB.
                    total_points = len(reshaped_coordinates[real_mask])
                    resampled_indices = torch.rand(total_points) < self._subsample_prob
                    self._label_xyz.append(
                        reshaped_coordinates[real_mask][resampled_indices]
                    )
                    self._label_rgb.append(
                        reshaped_rgb[real_mask_rect][resampled_indices]
                    )
                    self._label_weight.append(
                        torch.ones(total_points)[resampled_indices] * pred_score
                    )
                    self._image_features.append(
                        einops.repeat(feature, "d -> b d", b=total_points)[
                            resampled_indices
                        ]
                    )
                    label_idx += 1
                    
        del self._model

        # Now, we summerize xyz coordinates, rgb values, confidence scores, and image features for each point.
        self._label_xyz = torch.cat(self._label_xyz).float()
        self._label_rgb = torch.cat(self._label_rgb).float()
        self._label_weight = torch.cat(self._label_weight).float()
        self._image_features = torch.cat(self._image_features).float()

    def _reshape_coordinates_and_get_valid(self, coordinates, data_dict):
        if "conf" in data_dict:
            # Real world data, find valid mask
            valid_mask = (
                torch.as_tensor(
                    (~np.isnan(data_dict["depth"]) & (data_dict["conf"] == 2))
                    & (data_dict["depth"] < 3.0)
                )
                .squeeze(0)
                .bool()
            )
            reshaped_coordinates = torch.as_tensor(coordinates)
            return reshaped_coordinates, valid_mask
        else:
            reshaped_coordinates = einops.rearrange(coordinates, "c h w -> (h w) c")
            valid_mask = torch.ones_like(coordinates).mean(dim=0).bool()
            return reshaped_coordinates, valid_mask

    def __getitem__(self, idx):
        # Create a dictionary with all relevant results.
        return {
            "xyz": self._label_xyz[idx].float(),
            "rgb": self._label_rgb[idx].float(),
            "clip_image_vector": self._image_features[idx].float(),
            "semantic_weight": self._label_weight[idx].float(),
        }

    def __len__(self):
        return len(self._label_xyz)

    @staticmethod
    def process_text(x: str) -> str:
        return x.replace("-", " ").replace("_", " ").lstrip().rstrip().lower()
    
    def _setup_owl_all_classes(self, view_data: R3DSemanticDataset):
        # Unifying all the class labels. 
        prebuilt_class_names = [
            OWLViTLabelledDataset.process_text(x)
            for x in view_data._id_to_name.values()
        ]
        
        # We find a photo of prompt can improve OWL-ViT's performance
        self._all_classes = ['a photo of ' + class_name for class_name in prebuilt_class_names]
