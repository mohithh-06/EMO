import os
import json
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from tagging import tag_closet_item

# Absolute path to test image
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_IMAGE = os.path.join(BASE_DIR, "Images", "shorts.jpg")

# Run tagging
result = tag_closet_item(TEST_IMAGE)

# Display result
print("\n[INFO] Tagging Result:")
print(json.dumps(result, indent=2))

# Save JSON to Closet folder if successful
if "error" not in result:
    CLOSET_DIR = os.path.join(BASE_DIR, "Closet")
    os.makedirs(CLOSET_DIR, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(TEST_IMAGE))[0]
    out_path = os.path.join(CLOSET_DIR, f"{base_name}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n[OK] Saved to {out_path}")
else:
    print("\n[ERROR] Tagging failed")
