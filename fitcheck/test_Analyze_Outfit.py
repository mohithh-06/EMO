"""
Test script for the analyze_outfit function in emo.
Loads an image from the 'Images' folder and prints the structured critique.
"""
import os
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(__file__))
from analyze_outfit import analyze_outfit_pipeline

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
test_img_path = os.path.join(BASE_DIR, "Images", "fit_look_01.jpg")

print(f"[DEBUG] Calling analyze_outfit_pipeline with '{test_img_path}'")
result = analyze_outfit_pipeline(test_img_path)

print("\n--- EMO EDITORIAL REPORT ---")
print(f"Headline: {result['critique']['headline']}")
print(f"Overall Score: {result['overall_score']}/100 ({result['tier_title']})")
print(f"Gender Detected: {result['vision']['gender']['label']} ({result['vision']['gender']['confidence']}%)")
print(f"Garment Detected: {result['vision']['garments']['primary']['label']}")
print(f"Dominant Palette: {[c['name'] for c in result['palette']]}")
print(f"Harmony: {result['harmony']['harmony_type']} (Contrast: {result['harmony']['contrast_ratio']}:1)")
print("\n--- SAVAGE VERDICT ---")
print(f"\"{result['critique']['savage_verdict']}\"")
print("\n--- 3 STYLING PIVOTS ---")
for p in result['pivots']:
    print(f"- [{p['category']}] {p['title']}: {p['action']}")