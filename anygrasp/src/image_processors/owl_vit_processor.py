from typing import List, Type

import torch
import numpy as np
from PIL import Image

from image_processors.image_processor import ImageProcessor
from transformers import OwlViTProcessor, OwlViTForObjectDetection

class OwlVITProcessor(ImageProcessor):
    def __init__(self):
        super().__init__()

        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    
    def detect_obj(
        self,
        image: Type[Image.Image],
        text: str = None,
        bbox: List[int] = None,
        visualize_box: bool = False,
        bbox_save_filename: str = None,
        visualize_boxes: bool = False,
        bboxes_save_filename: str = None,
    ) -> List[int] :
        texts = [[text, "A photo of " + text]]  
        inputs = self.processor(text=texts, images=image, return_tensors="pt")

        outputs = self.model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.01)

        text = texts[0]
        boxes, scores, _ = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        max_ind = np.argmax(scores.detach().numpy())
        max_box = boxes.detach().numpy()[max_ind].astype(int)

        if visualize_boxes: 
            self.draw_bounding_boxes(image, boxes, scores, max_ind, bboxes_save_filename)
        
        if visualize_box:
            self.draw_bounding_box(image, max_box, bbox_save_filename)

        return max_box

