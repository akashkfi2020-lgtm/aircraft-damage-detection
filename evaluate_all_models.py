"""
evaluate_all_models.py
Place this file inside:  /home/akash/Project Exhibition/

Run with:
    cd "/home/akash/Project Exhibition"
    python evaluate_all_models.py

What it does:
    - Combines train + valid + test images into a temporary dataset
    - Runs validation for each of the 4 trained models on the FULL dataset
    - Prints a per-class metrics table (like YOLO's console output) for each model
    - Renders and saves a styled PNG table for each model
    - Saves a combined summary CSV
"""

import os
import shutil
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = "/home/akash/Project Exhibition"
TEMP_DIR = os.path.join(BASE, "_full_dataset_temp")   # created & deleted automatically

MODELS = {
    "YOLOv8":  os.path.join(BASE, "YOLOv8/runs/detect/yolov8_aircraft_test/weights/best.pt"),
    "YOLOv10": os.path.join(BASE, "YOLOv10/runs/detect/yolov10_aircraft_finetune/weights/best.pt"),
    "YOLOv11": os.path.join(BASE, "YOLOv11/runs/detect/yolo11_aircraft_finetune/weights/best.pt"),
    "RT-DETR": os.path.join(BASE, "RT-DETR/runs/detect/rtdetr_aircraft_finetune/weights/best.pt"),
}

# Use YOLOv8's dataset as the reference (all models share the same images/labels)
DATASET_ROOT = os.path.join(BASE,
    "YOLOv8/aircraft-skin-defects.v4-classification-isolated-5-classes-grayscale-prep.yolov8")

SPLITS = ["train", "valid", "test"]


# ── Step 1 : Build a merged dataset folder ────────────────────────────────────
def build_full_dataset():
    """Merge train/valid/test into a single 'valid' split YOLO can evaluate."""
    merged_img = os.path.join(TEMP_DIR, "images")
    merged_lbl = os.path.join(TEMP_DIR, "labels")
    os.makedirs(merged_img, exist_ok=True)
    os.makedirs(merged_lbl, exist_ok=True)

    total = 0
    for split in SPLITS:
        img_dir = os.path.join(DATASET_ROOT, split, "images")
        lbl_dir = os.path.join(DATASET_ROOT, split, "labels")
        if not os.path.isdir(img_dir):
            print(f"  [skip] {split}/images not found")
            continue
        for fname in os.listdir(img_dir):
            src_img = os.path.join(img_dir, fname)
            stem    = os.path.splitext(fname)[0]
            src_lbl = os.path.join(lbl_dir, stem + ".txt")
            dst_img = os.path.join(merged_img, f"{split}_{fname}")
            dst_lbl = os.path.join(merged_lbl, f"{split}_{stem}.txt")
            shutil.copy2(src_img, dst_img)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)
            total += 1

    # Read class names from original yaml
    orig_yaml = os.path.join(DATASET_ROOT, "data.yaml")
    with open(orig_yaml) as f:
        orig_cfg = yaml.safe_load(f)

    # Write a minimal yaml pointing to the merged folder
    merged_yaml = os.path.join(TEMP_DIR, "full_data.yaml")
    with open(merged_yaml, "w") as f:
        yaml.dump({
            "path": TEMP_DIR,
            "val":  "images",          # YOLO evaluates the 'val' key
            "nc":   orig_cfg["nc"],
            "names": orig_cfg["names"],
        }, f)

    print(f"\n✅ Merged dataset ready: {total} images → {TEMP_DIR}")
    return merged_yaml, orig_cfg["names"]


# ── Step 2 : Run validation & collect per-class metrics ──────────────────────
def evaluate_model(model_name, model_path, yaml_path):
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*60}")
    model   = YOLO(model_path)
    metrics = model.val(data=yaml_path, split="val", verbose=True)

    # metrics.ap_class_index → class indices evaluated
    # metrics.box.ap50       → per-class AP@50
    # metrics.box.ap         → per-class AP@50-95
    # metrics.box.p / .r     → per-class precision / recall (last threshold)

    class_indices = metrics.ap_class_index.tolist()
    rows = []
    for i, cls_idx in enumerate(class_indices):
        rows.append({
            "Class":      names[cls_idx],
            "Precision":  round(float(metrics.box.p[i]),  4),
            "Recall":     round(float(metrics.box.r[i]),  4),
            "mAP@50":     round(float(metrics.box.ap50[i]), 4),
            "mAP@50-95":  round(float(metrics.box.ap[i]),  4),
        })

    # Add overall 'all' row
    rows.insert(0, {
        "Class":     "all",
        "Precision": round(float(metrics.box.mp),    4),
        "Recall":    round(float(metrics.box.mr),    4),
        "mAP@50":    round(float(metrics.box.map50), 4),
        "mAP@50-95": round(float(metrics.box.map),   4),
    })

    return pd.DataFrame(rows)


# ── Step 3 : Render a styled table PNG for each model ────────────────────────
def render_table(df, model_name, out_path):
    metric_cols = ["Precision", "Recall", "mAP@50", "mAP@50-95"]
    n_rows      = len(df)

    fig, ax = plt.subplots(figsize=(10, 0.55 * n_rows + 1.2))
    ax.axis("off")

    table = ax.table(
        cellText  = df.values.tolist(),
        colLabels = list(df.columns),
        cellLoc   = "center",
        loc       = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 2.0)

    HEADER_BG  = "#1f4e79"
    ROW_COLORS = ["#1a1a2e", "#16213e"]
    BEST_COLOR = "#1a472a"
    ALL_COLOR  = "#3b2f4a"   # distinct colour for the 'all' row

    # Best per metric (excluding 'all' row which is index 0)
    sub = df.iloc[1:]
    best_idx = {c: sub[c].idxmax() for c in metric_cols if c in df.columns}

    col_names = list(df.columns)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#444466")
        cell.set_linewidth(0.5)
        if row == 0:                          # header
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            data_row = row - 1               # 0-indexed into df
            col_name = col_names[col]
            is_all   = (data_row == 0)       # 'all' row
            is_best  = (col_name in metric_cols
                        and not is_all
                        and best_idx.get(col_name) == data_row + 1)  # +1 offset

            if is_all:
                cell.set_facecolor(ALL_COLOR)
                cell.set_text_props(color="white", fontweight="bold")
            elif is_best:
                cell.set_facecolor(BEST_COLOR)
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor(ROW_COLORS[data_row % 2])
                cell.set_text_props(color="white")

    fig.patch.set_facecolor("#0d0d1a")
    ax.set_title(f"{model_name} — Per-Class Metrics (Full Dataset)",
                 color="white", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📊 Table saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Build merged dataset
    merged_yaml, names = build_full_dataset()

    all_summary = []

    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"\n[SKIP] {model_name} — weights not found at:\n  {model_path}")
            continue

        # 2. Evaluate
        df = evaluate_model(model_name, model_path, merged_yaml)

        # 3. Print table to console (clean)
        print(f"\n--- {model_name} : Per-Class Results (Full Dataset) ---")
        print(df.to_string(index=False))

        # 4. Save styled PNG table
        table_path = os.path.join(BASE, f"{model_name}_full_dataset_metrics.png")
        render_table(df, model_name, table_path)

        # 5. Collect 'all' row for summary
        all_row = df[df["Class"] == "all"].copy()
        all_row.insert(0, "Model", model_name)
        all_summary.append(all_row)

    # 6. Save combined summary CSV
    summary_df = pd.concat(all_summary, ignore_index=True)
    summary_csv = os.path.join(BASE, "full_dataset_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✅ Summary CSV saved → {summary_csv}")
    print(summary_df.to_string(index=False))

    # 7. Cleanup temp folder
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print(f"\n🧹 Temp folder removed: {TEMP_DIR}")
    print("\n✅ Done.")
