from typing import List, Type

from PIL import Image
import numpy as np

from image_processors.image_processor import ImageProcessor
from segment_anything import sam_model_registry, SamPredictor

class SamProcessor(ImageProcessor):
    def __init__(self):
        super().__init__()
        
        sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device)

        self.model = SamPredictor(sam)
    
    def detect_obj(
        self,
        image: Type[Image.Image],
        text: str = None,
        bbox: List[int] = None,
        save_file: str = None,
    ) -> np.ndarray:

        self.model.set_image(image)
        masks, _, _ = self.model.predict(
            point_coords = None,
            point_labels = None,
            box = bbox,
            multimask_output = False
        )

        return masks
