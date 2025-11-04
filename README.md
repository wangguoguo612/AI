# QDCC-AI-Optimization
Artificial Intelligence for Quantum Dot Color Conversion Layer Optimization in Full-Color Micro-LED Displays

This project explores AI-assisted optimization of Quantum Dot Color Conversion (QDCC) layers for full-color Micro-LED displays.
It applies machine learning and deep learning techniques for process parameter optimization, optical performance prediction, and defect detection.

## Structure
```
QDCC-AI-Optimization/
│
├── README.md
├── data/
│   ├── qdcc_dataset.csv
│   ├── defect_images.npz
│   └── defect_samples/
├── models/
│   ├── svr_cie_x.pkl
│   ├── rf_R_intensity.pkl
│   └── scaler.pkl
├── src/
│   ├── generate_dataset.py
│   ├── train_models.py
│   ├── defect_dataset_gen.py
│   └── predict.py
└── results/
    └── figures/
```

## Installation
```
pip install numpy pandas scikit-learn matplotlib pillow tensorflow
```

## Quick Start
```
python src/generate_dataset.py
python src/train_models.py
python src/predict.py --voltage 80 --conc 1.2 --speed 2000 --temp 100
```

## Results
SVR (CIE_x): R² ≈ 0.90  
RandomForest (R_intensity): R² ≈ 0.87
