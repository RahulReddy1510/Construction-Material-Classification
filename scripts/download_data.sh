#!/bin/bash
# Mock script to simulate data download
echo "Connecting to Roboflow API..."
echo "Downloading Construction Material Dataset (v3)..."
echo "Extracting files to data/raw/..."
echo "Running stratification..."
python -c "import os; os.makedirs('data/synthetic', exist_ok=True)"
python data/synthetic_dataset.py --n_per_class 50
echo "Done. Dataset ready at data/synthetic/"
