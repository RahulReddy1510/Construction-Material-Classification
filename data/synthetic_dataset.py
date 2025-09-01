import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from scipy.spatial import Voronoi
import random

def generate_texture_image(material_class, size=(224, 224), seed=None):
    """
    Generates a visually distinct procedural texture for a material class.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    height, width = size
    img = np.zeros((height, width, 3), dtype=np.uint8)

    if material_class == "concrete":
        # Gray base + noise + streaks
        base_color = np.array([136, 136, 136]) # #888888
        img[:] = base_color + np.random.normal(0, 15, (height, width, 3)).astype(np.int16).clip(-30, 30)
        # Add horizontal streaks
        streak_mask = np.random.random((height, width)) > 0.98
        img[streak_mask] = [100, 100, 100]
        img = cv2.GaussianBlur(img, (7, 1), 0) # Horizontal blur for streaks

    elif material_class == "brick":
        # Reddish-brown base
        img[:] = [48, 64, 160] # BGR for #A04030
        mortar_color = [176, 192, 208] # BGR for #D0C0B0
        
        brick_h, brick_w = 40, 80
        mortar_thickness = 3
        
        for y in range(0, height, brick_h + mortar_thickness):
            offset = (brick_w // 2) if (y // (brick_h + mortar_thickness)) % 2 == 1 else 0
            # Draw horizontal mortar lines
            cv2.line(img, (0, y), (width, y), mortar_color, mortar_thickness)
            for x in range(-offset, width, brick_w + mortar_thickness):
                # Draw vertical mortar lines
                cv2.line(img, (x, y), (x, y + brick_h), mortar_color, mortar_thickness)
        
        # Add some noise to make it less perfect
        noise = np.random.normal(0, 10, (height, width, 3)).astype(np.int16)
        img = (img.astype(np.int16) + noise).clip(0, 255).astype(np.uint8)

    elif material_class == "metal":
        # Silver/Blue-gray base
        img[:] = [208, 200, 192] # BGR for #C0C8D0
        # Directional grain
        for _ in range(100):
            y = random.randint(0, height-1)
            cv2.line(img, (0, y), (width, y), [180, 170, 160], 1)
        # Specular highlight (top-left)
        center = (width//4, height//4)
        axes = (width//2, height//2)
        overlay = img.copy()
        cv2.ellipse(overlay, center, axes, 0, 0, 360, [255, 255, 255], -1)
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
        # High frequency noise
        img = (img.astype(np.int16) + np.random.normal(0, 5, (height, width, 3)).astype(np.int16)).clip(0, 255).astype(np.uint8)

    elif material_class == "wood":
        # Brown base
        img[:] = [60, 94, 139] # BGR for #8B5E3C
        # Sine wave grain
        x = np.linspace(0, 10 * np.pi, width)
        for y in range(height):
            # shift is an array of shape (width,)
            shift = (np.sin(x + y * 0.05) * 5).astype(np.int16)
            # Add shift to each pixel in the row (broadcasting across BGR channels)
            row = img[y].astype(np.int16) + shift[:, np.newaxis]
            img[y] = row.clip(0, 255).astype(np.uint8)
        # Knots
        for _ in range(2):
            kx, ky = random.randint(0, width), random.randint(0, height)
            cv2.circle(img, (kx, ky), random.randint(5, 15), [40, 60, 100], -1)
        img = cv2.GaussianBlur(img, (3, 3), 0)

    elif material_class == "stone":
        # Gray-brown base
        img[:] = [122, 138, 154] # BGR for #9A8A7A
        # Voronoi pattern
        points = np.random.randint(0, height, (30, 2))
        vor = Voronoi(points)
        for region in vor.regions:
            if not region or -1 in region: continue
            polygon = [vor.vertices[i] for i in region]
            poly_np = np.array(polygon, dtype=np.int32)
            color = [v + random.randint(-20, 20) for v in [122, 138, 154]]
            cv2.fillPoly(img, [poly_np], color)
        # Add rough edges
        img = cv2.Canny(img, 50, 150)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_base = np.full((height, width, 3), [122, 138, 154], dtype=np.uint8)
        img = cv2.addWeighted(img, 0.2, img_base, 0.8, 0)

    return img

def augment_synthetic(image):
    """Apply slight random variations to synthetic images."""
    # Random brightness
    alpha = 1.0 + random.uniform(-0.2, 0.2)
    # Random rotation
    center = (image.shape[1] // 2, image.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, random.uniform(-15, 15), 1.0)
    image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def generate_synthetic_dataset(output_dir, n_train=1200, n_val=200, n_test=200, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    classes = ["concrete", "brick", "metal", "wood", "stone"]
    splits = {"train": n_train, "val": n_val, "test": n_test}
    
    for split, count in splits.items():
        print(f"Generating {split} split...")
        for cls in classes:
            cls_dir = os.path.join(output_dir, split, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for i in tqdm(range(count), desc=cls):
                img = generate_texture_image(cls, seed=seed+i if seed else None)
                if split == "train":
                    img = augment_synthetic(img)
                cv2.imwrite(os.path.join(cls_dir, f"{cls}_{i:04d}.jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic construction material dataset")
    parser.add_argument("--output_dir", type=str, default="data/synthetic/", help="Output directory")
    parser.add_argument("--n_per_class", type=int, default=200, help="Number of images per class for training split")
    args = parser.parse_args()
    
    generate_synthetic_dataset(args.output_dir, n_train=args.n_per_class, n_val=50, n_test=50)
    print(f"Synthetic dataset generated at {args.output_dir}")
