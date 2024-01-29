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
from dataloaders.scannet_200_classes import SCANNET_COLOR_MAP_200, CLASS_LABELS_200


# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset

setup_logger()
d2_logger = logging.getLogger("detectron2")
d2_logger.setLevel(level=logging.WARNING)

# import some common libraries
import sys

# import some common detectron2 utilities
from detectron2.data import MetadataCatalog

import torchvision.transforms as transforms
from transformers import AutoProcessor, OwlViTForObjectDetection
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


#def get_clip_embeddings(vocabulary, prompt="a photo of "):
#    text_encoder = build_text_encoder(pretrain=True)
#    text_encoder.eval()
#    texts = [prompt + x.replace("-", " ") for x in vocabulary]
#    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
#    return emb


# New visualizer class to disable jitter.
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import matplotlib.colors as mplc

class LowJitterVisualizer(Visualizer):
    def _jitter(self, color):
        """
        Randomly modifies given color to produce a slightly different color than the color given.

        Args:
            color (tuple[double]): a tuple of 3 elements, containing the RGB values of the color
                picked. The values in the list are in the [0.0, 1.0] range.

        Returns:
            jittered_color (tuple[double]): a tuple of 3 elements, containing the RGB values of the
                color after being jittered. The values in the list are in the [0.0, 1.0] range.
        """
        color = mplc.to_rgb(color)
        vec = np.random.rand(3)
        # better to do it in another color space
        vec = vec / np.linalg.norm(vec)
        vec *= 0.01  # 1% noise in the color
        res = np.clip(vec + color, 0, 1)
        return tuple(res)


SCANNET_NAME_TO_COLOR = {
    x: np.array(c) for x, c in zip(CLASS_LABELS_200, SCANNET_COLOR_MAP_200.values())
}

SCANNET_ID_TO_COLOR = {
    i: np.array(c) for i, c in enumerate(SCANNET_COLOR_MAP_200.values())
}

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
    LSEG_LABEL_WEIGHT = 0.1
    LSEG_IMAGE_DISTANCE = 10.0

    def __init__(
        self,
        #view_dataset: Union[R3DSemanticDataset, Subset[R3DSemanticDataset]],
        view_dataset,
        owl_model_name: str = "google/owlvit-base-patch32",
        sentence_encoding_model_name="all-mpnet-base-v2",
        sam_model_type = "vit_h",
        sam_model_path_name="sam_vit_h_4b8939.pth",
        device: str = "cuda",
        batch_size: int = 1,
        threshold: float = 0.1,
        num_images_to_label: int = -1,
        subsample_prob: float = 0.2,
        use_extra_classes: bool = False,
        use_gt_classes: bool = True,
        exclude_gt_images: bool = False,
        gt_inst_images: Optional[List[int]] = None,
        gt_sem_images: Optional[List[int]] = None,
        visualize_results: bool = False,
        visualization_path: Optional[str] = None,
        use_scannet_colors: bool = True,
    ):
        dataset = view_dataset
        view_data = (
            view_dataset.dataset if isinstance(view_dataset, Subset) else view_dataset
        )
        self._image_width, self._image_height = view_data.image_size
        self._model = OwlViTForObjectDetection.from_pretrained(owl_model_name).to(device)
        self._processor = AutoProcessor.from_pretrained(owl_model_name)
        sentence_model = SentenceTransformer(sentence_encoding_model_name)

        self._batch_size = batch_size
        self._device = device
        self._owl_threshold = threshold
        self._subsample_prob = subsample_prob

        self._label_xyz = []
        self._label_rgb = []
        self._label_weight = []
        self._label_idx = []
        self._text_ids = []
        self._text_id_to_feature = {}
        self._image_features = []
        self._distance = []

        self._exclude_gt_image = exclude_gt_images
        images_to_label = self.get_best_sem_segmented_images(
            dataset, num_images_to_label, gt_inst_images, gt_sem_images
        )
        self._use_extra_classes = use_extra_classes
        self._use_gt_classes = use_gt_classes
        self._use_scannet_colors = use_scannet_colors

        self._visualize = visualize_results
        if self._visualize:
            assert visualization_path is not None
            self._visualization_path = Path(visualization_path)
            os.makedirs(self._visualization_path, exist_ok=True)
        # First, setup detic with the combined classes.
        self._setup_owl_all_classes(view_data)
        if not os.path.exists(sam_model_path_name):
            url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
            wget.download(url, out = sam_model_path_name)
        sam = sam_model_registry[sam_model_type](checkpoint=sam_model_path_name)
        mask_predictor = SamPredictor(sam)
        mask_predictor.model = mask_predictor.model.cuda()
        self._setup_owl_dense_labels(
            dataset, images_to_label, mask_predictor, sentence_model
        )

        del mask_predictor
        del sentence_model

    def get_best_sem_segmented_images(
        self,
        dataset,
        num_images_to_label: int,
        gt_inst_images: Optional[List[int]] = None,
        gt_sem_images: Optional[List[int]] = None,
    ):
        # Using depth as a proxy for object diversity in a scene.
        if self._exclude_gt_image:
            assert gt_inst_images is not None
            assert gt_sem_images is not None
        num_objects_and_images = []
        for idx in range(len(dataset)):
            if self._exclude_gt_image:
                if idx in gt_inst_images or idx in gt_sem_images:
                    continue
            num_objects_and_images.append(
                (dataset[idx]["depth"].max() - dataset[idx]["depth"].min(), idx)
            )

        sorted_num_object_and_img = sorted(
            num_objects_and_images, key=lambda x: x[0], reverse=True
        )
        return [x[1] for x in sorted_num_object_and_img[:num_images_to_label]]

    @torch.no_grad()
    def _setup_owl_dense_labels(
        self, dataset, images_to_label, mask_predictor, sentence_model
    ):
        # Now just iterate over the images and do Detic preprocessing.
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)
        label_idx = 0
        text_strings = [
            OWLViTLabelledDataset.process_text(x) for x in self._all_classes
        ]
        #log = 0
        for idx, data_dict in tqdm.tqdm(
            enumerate(dataloader), total=len(dataset), desc="Calculating OWL features"
        ):
            #print(log)
            if idx not in images_to_label:
                continue
            rgb = einops.rearrange(data_dict["rgb"][..., :3], "b h w c -> b c h w")
            xyz = data_dict["xyz_position"]
            owl_queries = ['a photo of ' + class_name for class_name in self._all_classes]
            for image, coordinates in zip(rgb, xyz):
                # Now calculate the Detic classification for this.
                #print(image.size())
                target_sizes = torch.Tensor([image[0].size()])
                #inputs = self._processor(text=self._all_classes, images=image, return_tensors="pt")
                inputs = self._processor(text=owl_queries, images=image, return_tensors="pt")
                for input in inputs:
                    inputs[input] = inputs[input].to(self._device)
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    results = post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=self._owl_threshold)
                    i = 0
                    text = self._all_classes[i]
                    boxes, scores, labels, features = results[i]["boxes"], results[i]["scores"], results[i]["labels"], results[i]['class_embed']
                # Now extract the results from the image and store them
                input_boxes = boxes.detach().to(mask_predictor.device) 
                #plt.imshow(image.permute(1, 2, 0))
                #print(image.shape)
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
                    #cv2.imwrite(str(self._visualization_path / f"Clean_{idx}.jpg"), image_vis)
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
                    #pred_mask = torch.full((img_h.int().item(), img_w.int().item()), False, dtype=torch.bool)
                    #pred_mask[boxes[i][0].int():boxes[i][2].int(), boxes[i][1].int():boxes[i][3].int()] = True
                    real_mask = pred_mask[valid_mask]
                    real_mask_rect = valid_mask & pred_mask
                    # Go over each instance and add it to the DB.
                    total_points = len(reshaped_coordinates[real_mask])
                    resampled_indices = torch.rand(total_points) < self._subsample_prob
                    self._label_xyz.append(
                        reshaped_coordinates[real_mask][resampled_indices]
                    )
                    #log += len(reshaped_coordinates[real_mask][resampled_indices])
                    self._label_rgb.append(
                        reshaped_rgb[real_mask_rect][resampled_indices]
                    )
                    self._text_ids.append(
                        torch.ones(total_points)[resampled_indices]
                        * self._new_class_to_old_class_mapping[pred_class.item()]
                    )
                    self._label_weight.append(
                        torch.ones(total_points)[resampled_indices] * pred_score
                    )
                    self._image_features.append(
                        einops.repeat(feature, "d -> b d", b=total_points)[
                            resampled_indices
                        ]
                    )
                    self._label_idx.append(
                        torch.ones(total_points)[resampled_indices] * label_idx
                    )
                    self._distance.append(torch.zeros(total_points)[resampled_indices])
                    label_idx += 1
                    
        del self._model
        
        # Now, get all the sentence encoding for all the labels.
        text_strings = [
            OWLViTLabelledDataset.process_text(x) for x in self._all_classes
        ]
        text_strings += self._all_classes
        with torch.no_grad():
            all_embedded_text = sentence_model.encode(text_strings)
            all_embedded_text = torch.from_numpy(all_embedded_text).float()

        for i, feature in enumerate(all_embedded_text):
            self._text_id_to_feature[i] = feature

        # Now, we map from label to text using this model.
        self._label_xyz = torch.cat(self._label_xyz).float()
        self._label_rgb = torch.cat(self._label_rgb).float()
        self._label_weight = torch.cat(self._label_weight).float()
        self._image_features = torch.cat(self._image_features).float()
        self._text_ids = torch.cat(self._text_ids).long()
        self._label_idx = torch.cat(self._label_idx).long()
        self._distance = torch.cat(self._distance).float()
        self._instance = (
            torch.ones_like(self._text_ids) * -1
        ).long()  # We don't have instance ID from this dataset.

    def _resample(self):
        resampled_indices = torch.rand(len(self._label_xyz)) < self._subsample_prob
        logging.info(
            f"Resampling dataset down from {len(self._label_xyz)} points to {resampled_indices.long().sum().item()} points."
        )
        self._label_xyz = self._label_xyz[resampled_indices]
        self._label_rgb = self._label_rgb[resampled_indices]
        self._label_weight = self._label_weight[resampled_indices]
        self._image_features = self._image_features[resampled_indices]
        self._text_ids = self._text_ids[resampled_indices]
        self._label_idx = self._label_idx[resampled_indices]
        self._distance = self._distance[resampled_indices]
        self._instance = self._instance[resampled_indices]

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
            "label": self._text_ids[idx].long(),
            "instance": self._instance[idx].long(),
            "img_idx": self._label_idx[idx].long(),
            "distance": self._distance[idx].float(),
            "clip_vector": self._text_id_to_feature.get(
                self._text_ids[idx].item()
            ).float(),
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
        #prompt = ['', 'red ', 'orange ', 'yellow ', 'green ', 'cyan ', 'blue ', 'magenta ',
        #    'purple ', 'white ', 'black ', 'grey ', 'pink ', 'brown ', 'beige ', 'teal ']
        prebuilt_class_set = (
            set(prebuilt_class_names) if self._use_gt_classes else set()
        )
        filtered_new_classes = (
            [x for x in CLASS_LABELS_200 if x not in prebuilt_class_set]
            if self._use_extra_classes
            else []
        )

        self._all_classes = prebuilt_class_names + filtered_new_classes
        #all_classes = []
        #for name in self._all_classes:
        #    for p in prompt:
        #        all_classes.append(p + name)
        #self._all_classes = all_classes

        if self._use_gt_classes:
            self._new_class_to_old_class_mapping = {
                x: x for x in range(len(self._all_classes))
            }
        else:
            # We are not using all classes, so we should map which new/extra class maps
            # to which old class.
            for class_idx, class_name in enumerate(self._all_classes):
                if class_name in prebuilt_class_set:
                    old_idx = prebuilt_class_names.index(class_name)
                else:
                    old_idx = len(prebuilt_class_names) + filtered_new_classes.index(
                        class_name
                    )
                self._new_class_to_old_class_mapping[class_idx] = old_idx

        self._all_classes = [
            OWLViTLabelledDataset.process_text(x) for x in self._all_classes
        ]
        new_metadata = MetadataCatalog.get("__unused")
        new_metadata.thing_classes = self._all_classes
        if self._use_scannet_colors:
            new_metadata.thing_colors = SCANNET_ID_TO_COLOR
        self.metadata = new_metadata
        # Reset visualization threshold
        output_score_threshold = self._owl_threshold
