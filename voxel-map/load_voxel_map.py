import logging
import os
import pprint
import random
from typing import Dict, Union

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from dataloaders import (
    R3DSemanticDataset,
    OWLViTLabelledDataset,
)

DEVICE = "cuda"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_real_dataset(cfg):
    if cfg.use_cache:
        location_train_dataset = torch.load(cfg.cache_path)
    else:
        view_dataset = R3DSemanticDataset(cfg.dataset_path, cfg.custom_labels, subsample_freq=cfg.sample_freq)
        if cfg.web_models.segmentation != 'owl':
            location_train_dataset = DeticDenseLabelledDataset(
                view_dataset,
                clip_model_name=cfg.web_models.clip,
                sentence_encoding_model_name=cfg.web_models.sentence,
                device=cfg.device,
                threshold=cfg.threshold,
                subsample_prob=cfg.subsample_prob,
                use_lseg=cfg.use_lseg,
                use_extra_classes=cfg.use_extra_classes,
                use_gt_classes=cfg.use_gt_classes_in_detic,
                visualize_results=cfg.visualize_detic_results,
                visualization_path=cfg.detic_visualization_path,
            )
        else:
            location_train_dataset = OWLViTLabelledDataset(
                view_dataset,
                owl_model_name=cfg.web_models.clip,
                #sentence_encoding_model_name=cfg.web_models.sentence,
                device=cfg.device,
                threshold=cfg.threshold,
                subsample_prob=cfg.subsample_prob,
                use_extra_classes=cfg.use_extra_classes,
                use_gt_classes=cfg.use_gt_classes_in_detic,
                visualize_results=cfg.visualize_detic_results,
                visualization_path=cfg.detic_visualization_path,
            )
    if cfg.cache_result:
        torch.save(location_train_dataset, cfg.cache_path)
    return location_train_dataset


@hydra.main(version_base="1.2", config_path="configs", config_name="train.yaml")
def main(cfg):
    # Set up single thread tokenizer.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    real_dataset = get_real_dataset(cfg)


if __name__ == "__main__":
    main()
