"""
PyTorch Multi-Task Deep Neural Network Architecture for emo Fashion AI.
Accepts 512-dimensional CLIP ViT-B/32 visual feature embeddings and predicts:
1. Gender Presentation (Menswear, Womenswear, Unisex)
2. Garment Taxonomy (10 distinct fashion categories)
3. Formality & Occasion (4 levels)
4. Fabric & Texture Dynamics (5 tactile profiles)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

GENDER_CLASSES = ["Menswear", "Womenswear", "Unisex"]

GARMENT_CLASSES = [
    "Casual T-Shirt / Top",
    "Formal Button-Down Shirt",
    "Tailored Blazer / Suit",
    "Ethnic Kurta Set / Traditional",
    "Hoodie / Sweatshirt",
    "Leather / Moto Jacket",
    "Dress / Gown",
    "Denim Jeans / Trousers",
    "Casual Shorts",
    "Flared / Pencil Skirt",
    "Footwear / Shoes",
    "Handbag / Accessories"
]

FORMALITY_CLASSES = [
    "Relaxed Streetwear / Casual",
    "Smart Casual / Modern Office",
    "Black Tie / High Formal",
    "Ceremonial / Festive Heritage"
]

TEXTURE_CLASSES = [
    "Soft Breathable Cotton Jersey",
    "Crisp Poplin / Structured Cotton",
    "Heavy Structured Wool / Denim",
    "Flowing Silk / Satin / Chiffon",
    "Supple Leather / Moto Suede"
]


class EmoFashionNeuralHead(nn.Module):
    """
    Multi-task classification head trained on multimodal visual embeddings.
    Uses residual MLP feature projection layers followed by specialized task heads with dropout
    and batch normalization for regularized, high-generalization inference.
    """
    def __init__(self, in_features: int = 512, hidden_dim: int = 384, dropout: float = 0.25):
        super(EmoFashionNeuralHead, self).__init__()

        # Layer 1: 512 -> 384
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.drop1 = nn.Dropout(dropout)

        # Layer 2: 384 -> 192 (with residual skip projection)
        latent_dim = hidden_dim // 2
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.bn2 = nn.BatchNorm1d(latent_dim)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.drop2 = nn.Dropout(dropout * 0.8)

        # Residual skip projection from in_features to latent_dim
        self.skip_proj = nn.Linear(in_features, latent_dim)

        # Task 1: Gender Classification Head
        self.head_gender = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, len(GENDER_CLASSES))
        )

        # Task 2: Garment Taxonomy Head (12 fashion classes)
        self.head_garment = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, len(GARMENT_CLASSES))
        )

        # Task 3: Formality Head (4 levels)
        self.head_formality = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, len(FORMALITY_CLASSES))
        )

        # Task 4: Fabric / Texture Head (5 tactile profiles)
        self.head_texture = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, len(TEXTURE_CLASSES))
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass producing raw logits for all 4 fashion tasks.
        """
        # Ensure 2D tensor (batch_size, in_features)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # L2-normalize input embeddings for numerical stability
        x_norm = F.normalize(x, p=2, dim=-1)

        # Shared residual backbone
        h = self.drop1(self.act1(self.bn1(self.fc1(x_norm))))
        h = self.drop2(self.act2(self.bn2(self.fc2(h))))
        skip = self.skip_proj(x_norm)
        latent = h + skip

        logits = {
            "gender": self.head_gender(latent),
            "garment": self.head_garment(latent),
            "formality": self.head_formality(latent),
            "texture": self.head_texture(latent)
        }
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Produce normalized softmax probabilities for all tasks.
        """
        logits = self.forward(x)
        return {
            task: F.softmax(out, dim=-1)
            for task, out in logits.items()
        }
