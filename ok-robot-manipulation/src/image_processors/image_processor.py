from typing import List, Type, Any
from abc import ABC, abstractmethod
import copy

from PIL import Image, ImageDraw
import numpy as np
import cv2
from utils.utils import draw_rectangle


class ImageProcessor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def detect_obj(
        self, image: Type[Image.Image], text: str = None, bbox: List[int] = None
    ) -> Any:
        pass

    def draw_bounding_box(
        self, image: Type[Image.Image], bbox: List[int], save_file: str = None
    ) -> None:
        new_image = copy.deepcopy(image)
        draw_rectangle(new_image, bbox)

        if save_file is not None:
            new_image.save(save_file)

    def draw_bounding_boxes(
        self,
        image: Type[Image.Image],
        bboxes: List[int],
        scores: List[int],
        max_box_ind: int = -1,
        save_file: str = None,
    ) -> None:
        if max_box_ind != -1:
            max_score = np.max(scores.detach().numpy())
            max_ind = np.argmax(scores.detach().numpy())
        max_box = bboxes.detach().numpy()[max_ind].astype(int)

        new_image = copy.deepcopy(image)
        img_drw = ImageDraw.Draw(new_image)
        img_drw.rectangle(
            [(max_box[0], max_box[1]), (max_box[2], max_box[3])], outline="green"
        )
        img_drw.text(
            (max_box[0], max_box[1]), str(round(max_score.item(), 3)), fill="green"
        )

        for box, score, label in zip(bboxes, scores):
            box = [int(i) for i in box.tolist()]
            if score == max_score:
                img_drw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red")
                img_drw.text(
                    (box[0], box[1]), str(round(max_score.item(), 3)), fill="red"
                )
            else:
                img_drw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="white")
        new_image.save(save_file)
        print(f"Saved Detection boxes at {save_file}")

    def draw_mask_on_image(
        self, image: Type[Image.Image], seg_mask: np.ndarray, save_file: str = None
    ) -> None:
        image = np.array(image)
        image[seg_mask] = image[seg_mask] * 0.2

        # overlay mask
        highlighted_color = [179, 210, 255]
        overlay_mask = np.zeros_like(image)
        overlay_mask[seg_mask] = highlighted_color

        # placing mask over image
        alpha = 0.6
        highlighted_image = cv2.addWeighted(overlay_mask, alpha, image, 1, 0)
        highlighted_image = cv2.cvtColor(highlighted_image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(save_file, highlighted_image)
        print(f"Saved Segmentation Mask at {save_file}")
