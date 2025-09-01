import os
import shutil
import random
import argparse
import torch
from collections import Counter

def organize_into_splits(raw_dir, output_dir, train_ratio=0.72, val_ratio=0.14, test_ratio=0.14, seed=42):
    """
    Organizes raw images into stratified train/val/test splits.
    """
    random.seed(seed)
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    stats = {"train": 0, "val": 0, "test": 0, "classes": classes}
    
    for cls in classes:
        cls_raw_dir = os.path.join(raw_dir, cls)
        images = [f for f in os.listdir(cls_raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        for split, split_imgs in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
            split_cls_dir = os.path.join(output_dir, split, cls)
            os.makedirs(split_cls_dir, exist_ok=True)
            for img_name in split_imgs:
                shutil.copy(os.path.join(cls_raw_dir, img_name), os.path.join(split_cls_dir, img_name))
            stats[split] += len(split_imgs)
            
    return stats

def compute_class_weights(data_dir):
    """
    Computes inverse-frequency weights for CrossEntropyLoss.
    """
    train_dir = os.path.join(data_dir, "train")
    classes = sorted(os.listdir(train_dir))
    counts = []
    for cls in classes:
        n = len(os.listdir(os.path.join(train_dir, cls)))
        counts.append(n)
    
    counts = torch.tensor(counts, dtype=torch.float32)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(classes)
    return weights

def verify_dataset(data_dir):
    """
    Checks if the dataset is properly structured and contains images.
    """
    if not os.path.exists(data_dir):
        return {"status": "error", "message": f"Directory {data_dir} does not exist"}
    
    report = {}
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            report[split] = "Missing"
            continue
            
        classes = os.listdir(split_dir)
        cls_counts = {cls: len(os.listdir(os.path.join(split_dir, cls))) for cls in classes}
        report[split] = cls_counts
        
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset preparation utilities")
    parser.add_argument("--raw_dir", type=str, help="Path to raw unsplit data")
    parser.add_argument("--output_dir", type=str, default="data/processed/", help="Path to save splits")
    args = parser.parse_args()
    
    if args.raw_dir:
        stats = organize_into_splits(args.raw_dir, args.output_dir)
        print("Dataset Stats:", stats)
    else:
        print("Please provide --raw_dir to run organization.")
