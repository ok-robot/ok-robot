from typing import List, Type, Tuple

from PIL import Image
import numpy as np

from image_processors.image_processor import ImageProcessor
from lang_sam import LangSAM


class LangSAMProcessor(ImageProcessor):
    def __init__(self):
        super().__init__()

        self.model = LangSAM()

    def detect_obj(
        self,
        image: Type[Image.Image],
        text: str = None,
        bbox: List[int] = None,
        visualize_box: bool = False,
        box_filename: str = None,
        visualize_mask: bool = False,
        mask_filename: str = None,
    ) -> Tuple[np.ndarray, List[int]]:
        masks, boxes, phrases, logits = self.model.predict(image, text)
        if len(masks) == 0:
            return masks, None

        seg_mask = np.array(masks[0])
        bbox = np.array(boxes[0], dtype=int)

        if visualize_box:
            self.draw_bounding_box(image, bbox, box_filename)

        if visualize_mask:
            self.draw_mask_on_image(image, seg_mask, mask_filename)

        return seg_mask, bbox
