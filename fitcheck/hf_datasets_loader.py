"""
Hugging Face Datasets Integration Engine for emo Fashion AI.
Connects authenticated Hugging Face Hub pipelines to:
1. formido/outfit_recomendation (3,193 expert instruction-tuning outfit pairings)
2. mvasil/polyvore-outfits (25,132 high-resolution fashion items & taxonomy)
"""

import os
import io
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from PIL import Image

# Ensure HF Token is read from environment
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


_FORMIDO_DF: Optional[pd.DataFrame] = None
_POLYVORE_DF: Optional[pd.DataFrame] = None
_POLYVORE_CATS: Optional[pd.DataFrame] = None


def load_formido_dataset() -> pd.DataFrame:
    """
    Load and cache formido/outfit_recomendation dataset (3,193 styling records).
    """
    global _FORMIDO_DF
    if _FORMIDO_DF is not None:
        return _FORMIDO_DF

    try:
        from huggingface_hub import hf_hub_download
        fpath = hf_hub_download(
            repo_id="formido/outfit_recomendation",
            filename="new_data.csv",
            repo_type="dataset"
        )
        _FORMIDO_DF = pd.read_csv(fpath)
        return _FORMIDO_DF
    except Exception as e:
        print(f"[HF Datasets] Error loading formido dataset via hub download: {e}")
        try:
            # Fallback to direct hf:// URI via pandas
            _FORMIDO_DF = pd.read_csv("hf://datasets/formido/outfit_recomendation/new_data.csv")
            return _FORMIDO_DF
        except Exception as e2:
            print(f"[HF Datasets] Fallback failed: {e2}")
            return pd.DataFrame()


def query_formido_recommendations(
    query_text: str = "",
    gender: str = "Any",
    formality: str = "Any",
    max_results: int = 4
) -> List[Dict[str, Any]]:
    """
    Query formido/outfit_recomendation for styling instructions matching the user profile.
    """
    df = load_formido_dataset()
    if df.empty:
        return []

    # Clean query keywords
    keywords = [k.strip().lower() for k in query_text.split() if len(k.strip()) > 2]
    
    if gender.lower() in ["men", "menswear", "man", "male"]:
        keywords.append("man")
    elif gender.lower() in ["women", "womenswear", "woman", "female"]:
        keywords.append("woman")

    if formality.lower() in ["formal", "black tie", "office"]:
        keywords.append("formal")
    elif formality.lower() in ["casual", "streetwear"]:
        keywords.append("casual")

    mask = pd.Series(True, index=df.index)

    if keywords:
        for kw in keywords[:3]:
            # Match in instruction or input text
            sub_mask = df["instruction"].str.contains(kw, case=False, na=False) | df["output"].str.contains(kw, case=False, na=False)
            if sub_mask.sum() > 0:
                mask = mask & sub_mask

    matches = df[mask]
    if matches.empty:
        matches = df.sample(min(max_results, len(df)), random_state=42)

    results = []
    for idx, row in matches.head(max_results).iterrows():
        instruction_text = str(row.get("instruction", "Bespoke styling profile"))
        output_text = str(row.get("output", ""))

        # Parse outfit blocks
        results.append({
            "id": idx,
            "instruction": instruction_text,
            "input": str(row.get("input", "")),
            "output": output_text
        })

    return results


def load_polyvore_categories() -> pd.DataFrame:
    """
    Load Polyvore categories hierarchy from categories.csv.
    """
    global _POLYVORE_CATS
    if _POLYVORE_CATS is not None:
        return _POLYVORE_CATS

    try:
        from huggingface_hub import hf_hub_download
        fpath = hf_hub_download(
            repo_id="mvasil/polyvore-outfits",
            filename="categories.csv",
            repo_type="dataset"
        )
        _POLYVORE_CATS = pd.read_csv(fpath)
        return _POLYVORE_CATS
    except Exception as e:
        print(f"[HF Datasets] Error loading polyvore categories: {e}")
        return pd.DataFrame()


def load_polyvore_samples(count: int = 6) -> List[Dict[str, Any]]:
    """
    Load sample items with high-res PIL images from mvasil/polyvore-outfits validation split.
    """
    try:
        from huggingface_hub import hf_hub_download
        fpath = hf_hub_download(
            repo_id="mvasil/polyvore-outfits",
            filename="data/nondisjoint/validation.parquet",
            repo_type="dataset"
        )
        df_val = pd.read_parquet(fpath)
        sample_rows = df_val.sample(min(count, len(df_val)), random_state=42)

        items = []
        for _, row in sample_rows.iterrows():
            item_id = str(row["item_id"])
            img_bytes = row["image"]["bytes"]
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            items.append({
                "item_id": item_id,
                "image": pil_img,
                "size": pil_img.size
            })
        return items
    except Exception as e:
        print(f"[HF Datasets] Error loading polyvore samples: {e}")
        return []
