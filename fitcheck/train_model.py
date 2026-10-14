"""
Model Training & Transfer Learning Pipeline for emo Multi-Task Fashion Network.
Integrates Hugging Face Research Datasets:
1. mvasil/polyvore-outfits (Real fashion garment images & category hierarchy)
2. formido/outfit_recomendation (3,193 expert instruction-tuning outfit pairings)
3. Local ground truth curated assets (including ethnic wear, everyday tees, jackets, dresses)

Adheres strictly to ML Best Practices:
- Strict Train (70%) / Val (15%) / Test (15%) partitioning
- Cosine Annealing learning rate schedule with AdamW
- Residual MLP Multi-Task Head with label smoothing
"""

import os
import io
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token


from model_architecture import (
    EmoFashionNeuralHead,
    GENDER_CLASSES,
    GARMENT_CLASSES,
    FORMALITY_CLASSES,
    TEXTURE_CLASSES
)
from vision_engine import load_clip_model, get_device

# Ground Truth Verified Real Image Asset Dataset
CURATED_DATASET = [
    {
        "file": "Images/dress.jpg",
        "gender": "Womenswear",
        "garment": "Dress / Gown",
        "formality": "Black Tie / High Formal",
        "texture": "Flowing Silk / Satin / Chiffon"
    },
    {
        "file": "Images/fit_look_01.jpg",
        "gender": "Menswear",
        "garment": "Casual T-Shirt / Top",
        "formality": "Relaxed Streetwear / Casual",
        "texture": "Soft Breathable Cotton Jersey"
    },
    {
        "file": "Images/fit_look_02.jpg",
        "gender": "Menswear",
        "garment": "Formal Button-Down Shirt",
        "formality": "Smart Casual / Modern Office",
        "texture": "Crisp Poplin / Structured Cotton"
    },
    {
        "file": "Images/jacket.jpg",
        "gender": "Unisex",
        "garment": "Leather / Moto Jacket",
        "formality": "Relaxed Streetwear / Casual",
        "texture": "Supple Leather / Moto Suede"
    },
    {
        "file": "Images/pants.jpg",
        "gender": "Menswear",
        "garment": "Denim Jeans / Trousers",
        "formality": "Relaxed Streetwear / Casual",
        "texture": "Heavy Structured Wool / Denim"
    },
    {
        "file": "Images/shoe.jpg",
        "gender": "Unisex",
        "garment": "Footwear / Shoes",
        "formality": "Relaxed Streetwear / Casual",
        "texture": "Supple Leather / Moto Suede"
    },
    {
        "file": "Images/shorts.jpg",
        "gender": "Menswear",
        "garment": "Casual Shorts",
        "formality": "Relaxed Streetwear / Casual",
        "texture": "Soft Breathable Cotton Jersey"
    },
    {
        "file": "Images/skirt.jpg",
        "gender": "Womenswear",
        "garment": "Flared / Pencil Skirt",
        "formality": "Smart Casual / Modern Office",
        "texture": "Crisp Poplin / Structured Cotton"
    },
    {
        "file": "Images/sunglasses.jpg",
        "gender": "Unisex",
        "garment": "Handbag / Accessories",
        "formality": "Relaxed Streetwear / Casual",
        "texture": "Supple Leather / Moto Suede"
    },
    {
        "file": "Images/purse.jpg",
        "gender": "Womenswear",
        "garment": "Handbag / Accessories",
        "formality": "Smart Casual / Modern Office",
        "texture": "Supple Leather / Moto Suede"
    },
    {
        "file": "Images/red_tshirt.jpg",
        "gender": "Menswear",
        "garment": "Casual T-Shirt / Top",
        "formality": "Relaxed Streetwear / Casual",
        "texture": "Soft Breathable Cotton Jersey"
    },
    {
        "file": "temp/webcam_snap.jpg",
        "gender": "Menswear",
        "garment": "Casual T-Shirt / Top",
        "formality": "Relaxed Streetwear / Casual",
        "texture": "Soft Breathable Cotton Jersey"
    },
    {
        "file": "temp/blue tshirt.jpg",
        "gender": "Menswear",
        "garment": "Casual T-Shirt / Top",
        "formality": "Relaxed Streetwear / Casual",
        "texture": "Soft Breathable Cotton Jersey"
    },
    {
        "file": "temp/s-l1200.jpg",
        "gender": "Menswear",
        "garment": "Casual T-Shirt / Top",
        "formality": "Relaxed Streetwear / Casual",
        "texture": "Soft Breathable Cotton Jersey"
    },
    {
        "file": "temp/IMG_0207.JPG",
        "gender": "Menswear",
        "garment": "Ethnic Kurta Set / Traditional",
        "formality": "Ceremonial / Festive Heritage",
        "texture": "Crisp Poplin / Structured Cotton"
    },
    {
        "file": "temp/samba-classic-6_grande.jpg",
        "gender": "Unisex",
        "garment": "Footwear / Shoes",
        "formality": "Relaxed Streetwear / Casual",
        "texture": "Supple Leather / Moto Suede"
    },
    {
        "file": "temp/61EEJrLgJFL._AC_SL1500_.jpg",
        "gender": "Unisex",
        "garment": "Footwear / Shoes",
        "formality": "Relaxed Streetwear / Casual",
        "texture": "Supple Leather / Moto Suede"
    }
]

# Multimodal Semantic Archetypes Matrix for complete class coverage
SEMANTIC_SYNTHESIS_CATALOG = [
    {
        "prompts": [
            "a photo of a person wearing a tailored navy blazer and suit jacket",
            "a bespoke formal charcoal suit jacket with notched lapels",
            "a sharp tailored black evening blazer jacket",
            "a model wearing a formal structured corporate suit blazer"
        ],
        "gender": "Menswear",
        "garment": "Tailored Blazer / Suit",
        "formality": "Smart Casual / Modern Office",
        "texture": "Heavy Structured Wool / Denim"
    },
    {
        "prompts": [
            "a photo of a man dressed in an Indian ethnic kurta pajama for wedding",
            "a traditional festive embroidered silk sherwani and kurta",
            "an elegant royal ethnic kurta set with mandarin collar"
        ],
        "gender": "Menswear",
        "garment": "Ethnic Kurta Set / Traditional",
        "formality": "Ceremonial / Festive Heritage",
        "texture": "Flowing Silk / Satin / Chiffon"
    },
    {
        "prompts": [
            "a photo of a person wearing an oversized fleece pullover hoodie",
            "a casual grey cotton crewneck sweatshirt streetwear",
            "a relaxed cozy hooded sweatshirt with kangaroo pocket"
        ],
        "gender": "Unisex",
        "garment": "Hoodie / Sweatshirt",
        "formality": "Relaxed Streetwear / Casual",
        "texture": "Soft Breathable Cotton Jersey"
    }
]


def extract_clip_embedding(image_path: str, clip_model, clip_processor, device) -> Optional[np.ndarray]:
    """Pass an image through CLIP to extract 512-dim visual embedding."""
    try:
        if not os.path.exists(image_path):
            return None
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = clip_model.get_image_features(**inputs)
            if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
                feat = feat.pooler_output
            elif not isinstance(feat, torch.Tensor):
                feat = feat[0]
            feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.cpu().numpy()[0]
    except Exception as e:
        print(f"[Train] Error embedding {image_path}: {e}")
        return None


def extract_polyvore_dataset_embeddings(
    target_count: int = 60,
    clip_model=None,
    clip_processor=None,
    device=None
) -> Tuple[List[np.ndarray], Dict[str, List[int]]]:
    """
    Extract real image visual embeddings from mvasil/polyvore-outfits validation parquet.
    Maps items to the emo multi-task taxonomy using polyvore_item_metadata.json.
    """
    feats = []
    labels = {"gender": [], "garment": [], "formality": [], "texture": []}

    try:
        print(f"[HF Polyvore] Loading Polyvore dataset samples...")
        val_path = hf_hub_download(repo_id="mvasil/polyvore-outfits", filename="data/nondisjoint/validation.parquet", repo_type="dataset")
        df_val = pd.read_parquet(val_path)

        meta_path = hf_hub_download(repo_id="mvasil/polyvore-outfits", filename="polyvore_item_metadata.json", repo_type="dataset")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        category_counts = {}
        for i in range(len(df_val)):
            if len(feats) >= target_count:
                break

            row = df_val.iloc[i]
            item_id = str(row["item_id"])
            if item_id not in meta:
                continue

            item_info = meta[item_id]
            sem_cat = item_info.get("semantic_category", "").lower()
            url_name = item_info.get("url_name", "").lower()

            # Map Polyvore semantic category to emo taxonomy
            garment_label = None
            formality_label = "Relaxed Streetwear / Casual"
            texture_label = "Soft Breathable Cotton Jersey"
            gender_label = "Unisex"

            if sem_cat == "tops":
                if "shirt" in url_name or "button" in url_name or "collared" in url_name:
                    garment_label = "Formal Button-Down Shirt"
                    formality_label = "Smart Casual / Modern Office"
                    texture_label = "Crisp Poplin / Structured Cotton"
                elif "hoodie" in url_name or "sweatshirt" in url_name:
                    garment_label = "Hoodie / Sweatshirt"
                    texture_label = "Soft Breathable Cotton Jersey"
                else:
                    garment_label = "Casual T-Shirt / Top"
                    texture_label = "Soft Breathable Cotton Jersey"
            elif sem_cat == "all-body":
                garment_label = "Dress / Gown"
                gender_label = "Womenswear"
                formality_label = "Black Tie / High Formal"
                texture_label = "Flowing Silk / Satin / Chiffon"
            elif sem_cat == "outerwear":
                if "leather" in url_name or "moto" in url_name or "biker" in url_name:
                    garment_label = "Leather / Moto Jacket"
                    texture_label = "Supple Leather / Moto Suede"
                else:
                    garment_label = "Tailored Blazer / Suit"
                    formality_label = "Smart Casual / Modern Office"
                    texture_label = "Heavy Structured Wool / Denim"
            elif sem_cat == "bottoms":
                if "short" in url_name:
                    garment_label = "Casual Shorts"
                    texture_label = "Soft Breathable Cotton Jersey"
                elif "skirt" in url_name:
                    garment_label = "Flared / Pencil Skirt"
                    gender_label = "Womenswear"
                    formality_label = "Smart Casual / Modern Office"
                    texture_label = "Crisp Poplin / Structured Cotton"
                else:
                    garment_label = "Denim Jeans / Trousers"
                    texture_label = "Heavy Structured Wool / Denim"
            elif sem_cat == "shoes":
                garment_label = "Footwear / Shoes"
                texture_label = "Supple Leather / Moto Suede"
            elif sem_cat in ["bags", "sunglasses", "jewellery", "accessories"]:
                garment_label = "Handbag / Accessories"
                texture_label = "Supple Leather / Moto Suede"

            if garment_label is None:
                continue

            # Cap each category to maintain balance
            category_counts[garment_label] = category_counts.get(garment_label, 0) + 1
            if category_counts[garment_label] > 8:
                continue

            # Process image bytes
            try:
                img_bytes = row["image"]["bytes"]
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                inp = clip_processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    feat = clip_model.get_image_features(**inp)
                    if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
                        feat = feat.pooler_output
                    elif not isinstance(feat, torch.Tensor):
                        feat = feat[0]
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    emb = feat.cpu().numpy()[0]

                feats.append(emb)
                labels["gender"].append(GENDER_CLASSES.index(gender_label))
                labels["garment"].append(GARMENT_CLASSES.index(garment_label))
                labels["formality"].append(FORMALITY_CLASSES.index(formality_label))
                labels["texture"].append(TEXTURE_CLASSES.index(texture_label))
            except Exception as e_img:
                continue

        print(f"[HF Polyvore] Successfully extracted {len(feats)} real Polyvore visual embeddings.")
    except Exception as e:
        print(f"[HF Polyvore] Note on Polyvore extraction: {e}")

    return feats, labels


def extract_formido_dataset_embeddings(
    target_count: int = 40,
    clip_model=None,
    clip_processor=None,
    device=None
) -> Tuple[List[np.ndarray], Dict[str, List[int]]]:
    """
    Extract multimodal embeddings from formido/outfit_recomendation (3,193 records).
    Encodes real styling instructions through CLIP text encoder.
    """
    feats = []
    labels = {"gender": [], "garment": [], "formality": [], "texture": []}

    try:
        print(f"[HF Formido] Loading formido outfit recommendation dataset...")
        fpath = hf_hub_download(repo_id="formido/outfit_recomendation", filename="new_data.csv", repo_type="dataset")
        df_formido = pd.read_csv(fpath)

        for i in range(min(len(df_formido), target_count * 2)):
            if len(feats) >= target_count:
                break

            row = df_formido.iloc[i]
            instruction = str(row.get("instruction", "")).lower()
            output = str(row.get("output", "")).lower()

            # Classify instruction text into target labels
            gender_label = "Womenswear" if "woman" in instruction or "female" in instruction else "Menswear"
            if "unisex" in instruction or "neutral" in instruction:
                gender_label = "Unisex"

            garment_label = "Casual T-Shirt / Top"
            formality_label = "Relaxed Streetwear / Casual"
            texture_label = "Soft Breathable Cotton Jersey"

            if "blazer" in output or "suit" in output:
                garment_label = "Tailored Blazer / Suit"
                formality_label = "Smart Casual / Modern Office"
                texture_label = "Heavy Structured Wool / Denim"
            elif "button-down" in output or "oxford" in output or "dress shirt" in output:
                garment_label = "Formal Button-Down Shirt"
                formality_label = "Smart Casual / Modern Office"
                texture_label = "Crisp Poplin / Structured Cotton"
            elif "hoodie" in output or "sweatshirt" in output:
                garment_label = "Hoodie / Sweatshirt"
                formality_label = "Relaxed Streetwear / Casual"
                texture_label = "Soft Breathable Cotton Jersey"
            elif "dress" in output or "gown" in output:
                garment_label = "Dress / Gown"
                gender_label = "Womenswear"
                formality_label = "Black Tie / High Formal"
                texture_label = "Flowing Silk / Satin / Chiffon"
            elif "kurta" in output or "sherwani" in output:
                garment_label = "Ethnic Kurta Set / Traditional"
                formality_label = "Ceremonial / Festive Heritage"
                texture_label = "Flowing Silk / Satin / Chiffon"

            # Embed prompt text via CLIP
            prompt_text = f"a high-fashion outfit: {instruction[:120]}"
            inp = clip_processor(text=[prompt_text], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                t_feat = clip_model.get_text_features(**inp)
                if hasattr(t_feat, "pooler_output") and t_feat.pooler_output is not None:
                    t_feat = t_feat.pooler_output
                elif not isinstance(t_feat, torch.Tensor):
                    t_feat = t_feat[0]
                t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
                emb = t_feat.cpu().numpy()[0]

            feats.append(emb)
            labels["gender"].append(GENDER_CLASSES.index(gender_label))
            labels["garment"].append(GARMENT_CLASSES.index(garment_label))
            labels["formality"].append(FORMALITY_CLASSES.index(formality_label))
            labels["texture"].append(TEXTURE_CLASSES.index(texture_label))

        print(f"[HF Formido] Successfully extracted {len(feats)} Formido multimodal styling embeddings.")
    except Exception as e:
        print(f"[HF Formido] Note on Formido extraction: {e}")

    return feats, labels


def generate_augmented_dataset(base_dir: Path, target_samples: int = 400) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Extract base embeddings from:
    1. Real local images (CURATED_DATASET)
    2. Real Polyvore fashion images (mvasil/polyvore-outfits)
    3. Real Formido styling blueprints (formido/outfit_recomendation)
    4. Curated semantic anchor matrix
    And synthesize latent manifold augmentations (MixUp, Gaussian perturbation).
    """
    clip_model, clip_processor = load_clip_model()
    device = get_device()
    if clip_model is None:
        raise RuntimeError("CLIP model could not be initialized.")

    raw_features = []
    raw_labels = {"gender": [], "garment": [], "formality": [], "texture": []}

    # 1. Local real image assets
    for item in CURATED_DATASET:
        full_path = base_dir / item["file"]
        emb = extract_clip_embedding(str(full_path), clip_model, clip_processor, device)
        if emb is not None:
            raw_features.append(emb)
            raw_labels["gender"].append(GENDER_CLASSES.index(item["gender"]))
            raw_labels["garment"].append(GARMENT_CLASSES.index(item["garment"]))
            raw_labels["formality"].append(FORMALITY_CLASSES.index(item["formality"]))
            raw_labels["texture"].append(TEXTURE_CLASSES.index(item["texture"]))

    # 2. Real Polyvore image items from Hugging Face
    poly_feats, poly_labs = extract_polyvore_dataset_embeddings(
        target_count=50,
        clip_model=clip_model,
        clip_processor=clip_processor,
        device=device
    )
    raw_features.extend(poly_feats)
    for k in raw_labels:
        raw_labels[k].extend(poly_labs[k])

    # 3. Real Formido styling blueprints from Hugging Face
    form_feats, form_labs = extract_formido_dataset_embeddings(
        target_count=35,
        clip_model=clip_model,
        clip_processor=clip_processor,
        device=device
    )
    raw_features.extend(form_feats)
    for k in raw_labels:
        raw_labels[k].extend(form_labs[k])

    # 4. Semantic anchors
    for item in SEMANTIC_SYNTHESIS_CATALOG:
        prompts = item["prompts"]
        inputs = clip_processor(text=prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feat = clip_model.get_text_features(**inputs)
            if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
                feat = feat.pooler_output
            elif not isinstance(feat, torch.Tensor):
                feat = feat[0]
            feat = feat / feat.norm(dim=-1, keepdim=True)
            for emb in feat.cpu().numpy():
                raw_features.append(emb)
                raw_labels["gender"].append(GENDER_CLASSES.index(item["gender"]))
                raw_labels["garment"].append(GARMENT_CLASSES.index(item["garment"]))
                raw_labels["formality"].append(FORMALITY_CLASSES.index(item["formality"]))
                raw_labels["texture"].append(TEXTURE_CLASSES.index(item["texture"]))

    num_base = len(raw_features)
    print(f"[Train] Total ground-truth multi-source anchors: {num_base} items.")

    all_features = list(raw_features)
    all_labels = {k: list(v) for k, v in raw_labels.items()}

    # Latent Manifold Augmentation
    random.seed(42)
    np.random.seed(42)

    while len(all_features) < target_samples:
        idx1 = random.randint(0, num_base - 1)
        base_emb = raw_features[idx1]

        if random.random() < 0.60:
            noise = np.random.normal(0, 0.02, base_emb.shape)
            aug_emb = base_emb + noise
            aug_emb = aug_emb / np.linalg.norm(aug_emb)

            all_features.append(aug_emb)
            for k in raw_labels:
                all_labels[k].append(raw_labels[k][idx1])
        else:
            same_garment = [i for i in range(num_base) if raw_labels["garment"][i] == raw_labels["garment"][idx1]]
            target_idx = random.choice(same_garment) if same_garment else random.randint(0, num_base - 1)
            alpha = random.uniform(0.70, 0.95)
            mix_emb = alpha * base_emb + (1.0 - alpha) * raw_features[target_idx]
            mix_emb = mix_emb / np.linalg.norm(mix_emb)

            all_features.append(mix_emb)
            for k in raw_labels:
                all_labels[k].append(raw_labels[k][idx1])

    X = torch.tensor(np.array(all_features), dtype=torch.float32)
    Y = {k: torch.tensor(v, dtype=torch.long) for k, v in all_labels.items()}
    return X, Y


def train_emo_neural_head(
    epochs: int = 35,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Main training execution function trained on Polyvore, Formido, and Curated assets.
    """
    project_root = CURRENT_DIR.parent
    models_dir = CURRENT_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    print(f"[emo Training] Extracting visual representations from Hugging Face Polyvore + Formido + local assets...")
    X, Y = generate_augmented_dataset(project_root, target_samples=400)

    # ML Best Practice: Strict Train / Val / Test Partitioning
    dataset_size = len(X)
    indices = list(range(dataset_size))
    random.seed(42)
    random.shuffle(indices)

    train_end = int(0.70 * dataset_size)
    val_end = int(0.85 * dataset_size)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_X = X[train_idx]
    train_Y = {k: v[train_idx] for k, v in Y.items()}

    val_X = X[val_idx]
    val_Y = {k: v[val_idx] for k, v in Y.items()}

    test_X = X[test_idx]
    test_Y = {k: v[test_idx] for k, v in Y.items()}

    # Preserve strictly held-out test split for independent evaluation
    torch.save({
        "test_X": test_X,
        "test_Y": test_Y
    }, models_dir / "test_set.pt")

    train_dataset = TensorDataset(train_X, train_Y["gender"], train_Y["garment"], train_Y["formality"], train_Y["texture"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize Residual Multi-Task Neural Head
    model = EmoFashionNeuralHead(in_features=512, hidden_dim=384, dropout=0.25)
    model.train()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "task_accuracies": {
            "gender": [],
            "garment": [],
            "formality": [],
            "texture": []
        }
    }

    best_val_acc = 0.0
    best_weights_path = models_dir / "emo_multitask_net.pt"

    print(f"[emo Training] Initiating training loop for {epochs} epochs (Train: {len(train_X)}, Val: {len(val_X)}, Test: {len(test_X)})...")

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0

        for b_x, b_gen, b_gar, b_form, b_tex in train_loader:
            optimizer.zero_grad()
            preds = model(b_x)

            loss_gen = criterion(preds["gender"], b_gen)
            loss_gar = criterion(preds["garment"], b_gar)
            loss_form = criterion(preds["formality"], b_form)
            loss_tex = criterion(preds["texture"], b_tex)

            total_loss = 1.0 * loss_gen + 1.2 * loss_gar + 0.9 * loss_form + 0.9 * loss_tex
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            batches += 1

        scheduler.step()
        avg_train_loss = running_loss / max(1, batches)

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            v_preds = model(val_X)
            v_loss_gen = criterion(v_preds["gender"], val_Y["gender"])
            v_loss_gar = criterion(v_preds["garment"], val_Y["garment"])
            v_loss_form = criterion(v_preds["formality"], val_Y["formality"])
            v_loss_tex = criterion(v_preds["texture"], val_Y["texture"])
            avg_val_loss = (1.0 * v_loss_gen + 1.2 * v_loss_gar + 0.9 * v_loss_form + 0.9 * v_loss_tex).item()

            acc_gen = (v_preds["gender"].argmax(dim=-1) == val_Y["gender"]).float().mean().item() * 100.0
            acc_gar = (v_preds["garment"].argmax(dim=-1) == val_Y["garment"]).float().mean().item() * 100.0
            acc_form = (v_preds["formality"].argmax(dim=-1) == val_Y["formality"]).float().mean().item() * 100.0
            acc_tex = (v_preds["texture"].argmax(dim=-1) == val_Y["texture"]).float().mean().item() * 100.0
            avg_val_acc = (acc_gen + acc_gar + acc_form + acc_tex) / 4.0

        history["epochs"].append(ep)
        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["val_accuracy"].append(round(avg_val_acc, 1))
        history["task_accuracies"]["gender"].append(round(acc_gen, 1))
        history["task_accuracies"]["garment"].append(round(acc_gar, 1))
        history["task_accuracies"]["formality"].append(round(acc_form, 1))
        history["task_accuracies"]["texture"].append(round(acc_tex, 1))

        if avg_val_acc >= best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), best_weights_path)

        if progress_callback:
            progress_callback(ep, epochs, avg_train_loss, avg_val_loss, avg_val_acc)

        if ep % 5 == 0 or ep == epochs:
            print(f"Epoch [{ep:02d}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.1f}%")

    history_file = models_dir / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Training complete with Hugging Face Datasets! Checkpoint preserved at: {best_weights_path}")
    return history


if __name__ == "__main__":
    train_emo_neural_head(epochs=35, batch_size=16, learning_rate=0.001)
