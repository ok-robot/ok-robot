import torch
import torch.nn.functional as F
import clip
from sentence_transformers import SentenceTransformer
from typing import List


class ClassificationExtractor:
    PROMPT = "A "
    EMPTY_CLASS = "Other"
    LOGIT_TEMP = 100.0

    def __init__(
        self,
        clip_model_name: str,
        sentence_model_name: str,
        class_names: List[str],
        device: str = "cuda",
        image_weight: float = 1.0,
        label_weight: float = 5.0,
    ):
        clip_model, _ = clip.load(clip_model_name, device=device)
        sentence_model = SentenceTransformer(sentence_model_name, device=device)

        # Adding this class in the beginning since the labels are 1-indexed.
        text_strings = []
        for name in class_names:
            text_strings.append(self.PROMPT + name.replace("-", " ").replace("_", " "))
        with torch.no_grad():
            all_embedded_text = sentence_model.encode(text_strings)
            all_embedded_text = torch.from_numpy(all_embedded_text).float().to(device)

        with torch.no_grad():
            text = clip.tokenize(text_strings).to(device)
            clip_encoded_text = clip_model.encode_text(text).float().to(device)

        del clip_model
        del sentence_model

        self.class_names = text_strings
        self.total_label_classes = len(text_strings)
        self._sentence_embed_size = all_embedded_text.size(-1)
        self._clip_embed_size = clip_encoded_text.size(-1)

        self._sentence_features = F.normalize(all_embedded_text, p=2, dim=-1)
        self._clip_text_features = F.normalize(clip_encoded_text, p=2, dim=-1)

        self._image_weight = image_weight
        self._label_weight = label_weight

    def calculate_classifications(
        self, model_text_features: torch.Tensor, model_image_features: torch.Tensor
    ):
        # Figure out the classification given the learned embedding of the objects.
        assert model_text_features.size(-1) == self._sentence_embed_size
        assert model_image_features.size(-1) == self._clip_embed_size

        # Now do the softmax over the classes.
        model_text_features = F.normalize(model_text_features, p=2, dim=-1)
        model_image_features = F.normalize(model_image_features, p=2, dim=-1)

        with torch.no_grad():
            text_logits = model_text_features @ self._sentence_features.T
            image_logits = model_image_features @ self._clip_text_features.T

        assert text_logits.size(-1) == self.total_label_classes
        assert image_logits.size(-1) == self.total_label_classes

        # Figure out weighted sum of probabilities.
        return (
            self._label_weight * F.softmax(self.LOGIT_TEMP * text_logits, dim=-1)
            + self._image_weight * F.softmax(self.LOGIT_TEMP * image_logits, dim=-1)
        ) / (self._label_weight + self._image_weight)
