import logging
from typing import List, Optional, Union
import clip
import einops
import os
import torch
import tqdm
import cv2

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
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor


DETIC_PATH = os.environ.get("DETIC_PATH", Path(__file__).parent / "../Detic")
LSEG_PATH = os.environ.get("LSEG_PATH", Path(__file__).parent / "../LSeg/")

sys.path.insert(0, f"{LSEG_PATH}/")
from encoding.models.sseg import BaseNet
from additional_utils.models import LSeg_MultiEvalModule
from modules.lseg_module import LSegModule
import torchvision.transforms as transforms

# Detic libraries
sys.path.insert(0, f"{DETIC_PATH}/third_party/CenterNet2/")
sys.path.insert(0, f"{DETIC_PATH}/")
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder

cfg = get_cfg()
add_centernet_config(cfg)
add_detic_config(cfg)
cfg.merge_from_file(
    f"{DETIC_PATH}/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
)
cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = (
    False  # For better visualization purpose. Set to False for all classes.
)
cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = (
    f"{DETIC_PATH}/datasets/metadata/lvis_v1_train_cat_info.json"
)
# cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.


def get_clip_embeddings(vocabulary, prompt="a "):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x.replace("-", " ") for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb


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


class DeticDenseLabelledDataset(Dataset):
    LSEG_LABEL_WEIGHT = 0.1
    LSEG_IMAGE_DISTANCE = 10.0

    def __init__(
        self,
        view_dataset: Union[R3DSemanticDataset, Subset[R3DSemanticDataset]],
        clip_model_name: str = "ViT-B/32",
        sentence_encoding_model_name="all-mpnet-base-v2",
        device: str = "cuda",
        batch_size: int = 1,
        threshold: float = 0.3,
        num_images_to_label: int = -1,
        subsample_prob: float = 0.2,
        use_lseg: bool = False,
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
        clip_model, _ = clip.load(clip_model_name, device=device)
        sentence_model = SentenceTransformer(sentence_encoding_model_name)

        self._batch_size = batch_size
        self._device = device
        self._detic_threshold = detic_threshold
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
        self._use_lseg = use_lseg
        self._use_extra_classes = use_extra_classes
        self._use_gt_classes = use_gt_classes
        self._use_scannet_colors = use_scannet_colors

        self._visualize = visualize_results
        if self._visualize:
            assert visualization_path is not None
            self._visualization_path = Path(visualization_path)
            os.makedirs(self._visualization_path, exist_ok=True)
        # First, setup detic with the combined classes.
        self._setup_detic_all_classes(view_data)
        self._setup_detic_dense_labels(
            dataset, images_to_label, clip_model, sentence_model
        )

        del clip_model
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
    def _setup_detic_dense_labels(
        self, dataset, images_to_label, clip_model, sentence_model
    ):
        # Now just iterate over the images and do Detic preprocessing.
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)
        label_idx = 0
        for idx, data_dict in tqdm.tqdm(
            enumerate(dataloader), total=len(dataset), desc="Calculating Detic features"
        ):
            if idx not in images_to_label:
                continue
            rgb = einops.rearrange(data_dict["rgb"][..., :3], "b h w c -> b c h w")
            xyz = data_dict["xyz_position"]
            for image, coordinates in zip(rgb, xyz):
                # Now calculate the Detic classification for this.
                with torch.no_grad():
                    result = self._predictor.model(
                        [
                            {
                                "image": image * 255,
                                "height": self._image_height,
                                "width": self._image_width,
                            }
                        ]
                    )[0]
                # Now extract the results from the image and store them
                instance = result["instances"]
                reshaped_rgb = einops.rearrange(image, "c h w -> h w c")
                (
                    reshaped_coordinates,
                    valid_mask,
                ) = self._reshape_coordinates_and_get_valid(coordinates, data_dict)
                if self._visualize:
                    v = LowJitterVisualizer(
                        reshaped_rgb,
                        self.metadata,
                        instance_mode=ColorMode.SEGMENTATION,
                    )
                    out = v.draw_instance_predictions(instance.to("cpu"))
                    cv2.imwrite(
                        str(self._visualization_path / f"{idx}.jpg"),
                        out.get_image()[:, :, ::-1],
                        [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                    )
                for pred_class, pred_mask, pred_score, feature in zip(
                    instance.pred_classes.cpu(),
                    instance.pred_masks.cpu(),
                    instance.scores.cpu(),
                    instance.features.cpu(),
                ):
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

        # First delete leftover Detic predictors
        del self._predictor

        if self._use_lseg:
            # Now, get to LSeg
            self._setup_lseg()
            for idx, data_dict in tqdm.tqdm(
                enumerate(dataloader),
                total=len(dataset),
                desc="Calculating LSeg features",
            ):
                if idx not in images_to_label:
                    continue
                rgb = einops.rearrange(data_dict["rgb"][..., :3], "b h w c -> b c h w")
                xyz = data_dict["xyz_position"]
                for image, coordinates in zip(rgb, xyz):
                    # Now figure out the LSeg lables.
                    with torch.no_grad():
                        unsqueezed_image = image.unsqueeze(0).float().cuda()
                        resized_image = self.resize(image).unsqueeze(0).cuda()
                        tfm_image = self.transform(unsqueezed_image)
                        outputs = self.evaluator.parallel_forward(
                            tfm_image, self._all_lseg_classes
                        )
                        image_feature = clip_model.encode_image(resized_image).squeeze(
                            0
                        )
                        image_feature = image_feature.cpu()
                        predicts = [torch.max(output, 1)[1].cpu() for output in outputs]
                    predict = predicts[0]

                    (
                        reshaped_coordinates,
                        valid_mask,
                    ) = self._reshape_coordinates_and_get_valid(coordinates, data_dict)
                    reshaped_rgb = einops.rearrange(image, "c h w -> h w c")

                    for label in range(len(self._all_classes)):
                        pred_mask = predict.squeeze(0) == label
                        real_mask = pred_mask[valid_mask]
                        real_mask_rect = valid_mask & pred_mask
                        total_points = len(reshaped_coordinates[real_mask])
                        resampled_indices = (
                            torch.rand(total_points) < self._subsample_prob
                        )
                        if total_points:
                            self._label_xyz.append(
                                reshaped_coordinates[real_mask][resampled_indices]
                            )
                            self._label_rgb.append(
                                reshaped_rgb[real_mask_rect][resampled_indices]
                            )
                            # Ideally, this should give all classes their true class label.
                            self._text_ids.append(
                                torch.ones(total_points)[resampled_indices]
                                * self._new_class_to_old_class_mapping[label]
                            )
                            # Uniform label confidence of LSEG_LABEL_WEIGHT
                            self._label_weight.append(
                                torch.ones(total_points)[resampled_indices]
                                * self.LSEG_LABEL_WEIGHT
                            )
                            self._image_features.append(
                                einops.repeat(
                                    image_feature, "d -> b d", b=total_points
                                )[resampled_indices]
                            )
                            self._label_idx.append(
                                torch.ones(total_points)[resampled_indices] * label_idx
                            )
                            self._distance.append(
                                torch.ones(total_points)[resampled_indices]
                                * self.LSEG_IMAGE_DISTANCE
                            )
                    # Since they all get the same image, here label idx is increased once
                    # at the very end.
                    label_idx += 1

            # Now, delete the module and the evaluator
            del self.evaluator
            del self.module
            del self.transform

        # Now, get all the sentence encoding for all the labels.
        text_strings = [
            DeticDenseLabelledDataset.process_text(x) for x in self._all_classes
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

    def _setup_detic_all_classes(self, view_data: R3DSemanticDataset):
        # Unifying all the class labels.
        predictor = DefaultPredictor(cfg)
        prebuilt_class_names = [
            DeticDenseLabelledDataset.process_text(x)
            for x in view_data._id_to_name.values()
        ]
        prebuilt_class_set = (
            set(prebuilt_class_names) if self._use_gt_classes else set()
        )
        filtered_new_classes = (
            [x for x in CLASS_LABELS_200 if x not in prebuilt_class_set]
            if self._use_extra_classes
            else []
        )

        self._all_classes = prebuilt_class_names + filtered_new_classes

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
            DeticDenseLabelledDataset.process_text(x) for x in self._all_classes
        ]
        new_metadata = MetadataCatalog.get("__unused")
        new_metadata.thing_classes = self._all_classes
        if self._use_scannet_colors:
            new_metadata.thing_colors = SCANNET_ID_TO_COLOR
        self.metadata = new_metadata
        classifier = get_clip_embeddings(new_metadata.thing_classes)
        num_classes = len(new_metadata.thing_classes)
        reset_cls_test(predictor.model, classifier, num_classes)
        # Reset visualization threshold
        output_score_threshold = self._detic_threshold
        for cascade_stages in range(len(predictor.model.roi_heads.box_predictor)):
            predictor.model.roi_heads.box_predictor[
                cascade_stages
            ].test_score_thresh = output_score_threshold
        self._predictor = predictor

    def find_in_class(self, classname):
        try:
            return self._all_classes.index(classname)
        except ValueError:
            ret_value = len(self._all_classes) + self._unfound_offset
            self._unfound_offset += 1
            return ret_value

    def _setup_lseg(self):
        self._lseg_classes = self._all_classes
        self._num_true_lseg_classes = len(self._lseg_classes)
        self._all_lseg_classes = self._all_classes  # + ["Other"]

        # We will try to classify all the classes, but will use LSeg labels for classes that
        # are not identified by Detic.
        LSEG_MODEL_PATH = f"{LSEG_PATH}/checkpoints/demo_e200.ckpt"
        try:
            self.module = LSegModule.load_from_checkpoint(
                checkpoint_path=LSEG_MODEL_PATH,
                data_path="",
                dataset="ade20k",
                backbone="clip_vitl16_384",
                aux=False,
                num_features=256,
                aux_weight=0,
                se_loss=False,
                se_weight=0,
                base_lr=0,
                batch_size=1,
                max_epochs=0,
                ignore_index=255,
                dropout=0.0,
                scale_inv=False,
                augment=False,
                no_batchnorm=False,
                widehead=True,
                widehead_hr=False,
                map_locatin=self._device,
                arch_option=0,
                block_depth=0,
                activation="lrelu",
            )
        except FileNotFoundError:
            LSEG_URL = "https://github.com/isl-org/lang-seg"
            raise FileNotFoundError(
                "LSeg model not found. Please download it from {} and place it in {}".format(
                    LSEG_URL, LSEG_MODEL_PATH
                )
            )
        if isinstance(self.module.net, BaseNet):
            model = self.module.net
        else:
            model = self.module

        model = model.eval()
        model = model.to(self._device)
        self.scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

        model.mean = [0.5, 0.5, 0.5]
        model.std = [0.5, 0.5, 0.5]

        self.transform = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.resize = transforms.Resize((224, 224))

        self.evaluator = LSeg_MultiEvalModule(model, scales=self.scales, flip=True).to(
            self._device
        )
        self.evaluator = self.evaluator.eval()
