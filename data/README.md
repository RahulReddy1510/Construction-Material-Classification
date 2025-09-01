# Data Pipeline

This directory contains all the logic for dataset acquisition, preparation, and augmentation.

## Contents

- `dataset_prep.py`: Utilities for downloading data from Roboflow/Kaggle and organizing it into stratified splits.
- `synthetic_dataset.py`: Procedural texture generation for testing the pipeline without a real dataset.
- `transforms.py`: Standardized PyTorch image transforms for training and validation.

## Materials Dataset

The project targets 5 classes of construction materials:
1. **Concrete**
2. **Brick**
3. **Metal**
4. **Wood**
5. **Stone**

### Sources
The primary dataset was assembled from:
- [Roboflow Universe](https://universe.roboflow.com/): Specifically construction material classification datasets.
- [Kaggle](https://www.kaggle.com/): Various construction site imagery.
- Deng et al. (2022) trustworthy CM dataset perspective.

### Split Strategy
We use a 72% / 14% / 14% split for Training, Validation, and Testing respectively.
- **Train**: ~6,000 images
- **Val**: ~1,000 images
- **Test**: ~1,000 images

## Synthetic Fallback

If you don't have the real dataset, you can generate a synthetic one for testing the code:

```bash
python data/synthetic_dataset.py --output_dir data/synthetic/ --n_per_class 200
```

The synthetic generator creates visually distinct textures:
- **Concrete**: Gray with streaks.
- **Brick**: Reddish-brown with mortar grid.
- **Metal**: Silver with directional grain and highlights.
- **Wood**: Brown with sine-wave grain and knots.
- **Stone**: Irregular Voronoi patterns.
