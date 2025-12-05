import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPModel,
    CLIPProcessor,
    CLIPTokenizer,
)

from config import GRIMConfig


class FeatureExtractor:
    def __init__(self):
        print("Loading DINOv2 and CLIP models (CPU)...")
        # Vision Model: DINOv2
        self.dino_processor = AutoImageProcessor.from_pretrained(GRIMConfig.DINO_MODEL)
        self.dino_model = AutoModel.from_pretrained(GRIMConfig.DINO_MODEL).to(
            GRIMConfig.DEVICE
        )

        # Text Model: CLIP
        self.clip_model = CLIPModel.from_pretrained(GRIMConfig.CLIP_MODEL).to(
            GRIMConfig.DEVICE
        )
        self.clip_processor = CLIPProcessor.from_pretrained(GRIMConfig.CLIP_MODEL)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(GRIMConfig.CLIP_MODEL)
        print("Models loaded.")

    def extract_task_embedding(self, task_description: str) -> torch.Tensor:
        """
        Encodes task text using CLIP.
        References:
            - Section III-B [cite: 143]
        """
        inputs = self.clip_tokenizer(
            [task_description], padding=True, return_tensors="pt"
        ).to(GRIMConfig.DEVICE)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def extract_dino_features(self, image: Image.Image):
        """
        Extracts dense DINOv2 descriptor fields.
        Returns:
            features (torch.Tensor): (1, n_patches, embed_dim)
            spatial_shape (tuple): (h_patches, w_patches)
        References:
            - Section III-C [cite: 125, 439]
        """
        inputs = self.dino_processor(images=image, return_tensors="pt").to(
            GRIMConfig.DEVICE
        )

        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        # Extract last hidden state
        # shape: (1, seq_len, hidden_size)
        last_hidden_state = outputs.last_hidden_state

        # Remove CLS token (index 0)
        patch_tokens = last_hidden_state[:, 1:, :]

        # Calculate spatial dimensions of the patches
        # DINOv2 usually uses patch size 14
        h, w = inputs["pixel_values"].shape[2], inputs["pixel_values"].shape[3]
        patch_h, patch_w = h // 14, w // 14

        return patch_tokens, (patch_h, patch_w)

    def interpolate_features_to_image(self, patch_tokens, spatial_shape, image_size):
        """
        Upsamples patch features to match original image resolution for pixel-wise mapping.
        """
        # Reshape to (1, embed_dim, H_patch, W_patch)
        features = patch_tokens.transpose(1, 2).reshape(
            1, -1, spatial_shape[0], spatial_shape[1]
        )

        # Bilinear interpolation to (H_img, W_img)
        features_upsampled = F.interpolate(
            features,
            size=image_size,  # (H, W) tuple
            mode="bilinear",
            align_corners=False,
        )

        # Return as (H*W, embed_dim)
        return (
            features_upsampled.squeeze(0)
            .permute(1, 2, 0)
            .reshape(-1, features.shape[1])
        )
