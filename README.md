
# Car Object Detection Project

## Overview
This project demonstrates how to fine-tune a pre-trained Faster R-CNN model (ResNet‐FPN backbone) using PyTorch’s `torchvision` library to detect cars in custom image data. It includes data preparation, model training, validation, inference, and a discussion of results.

## Dataset
- **Source:** [Kaggle - sshikamaru/car-object-detection](https://www.kaggle.com/datasets/sshikamaru/car-object-detection)
- **Structure:**
  ```
  data/
    training_images/            # Images used for training and validation
    testing_images/             # (Optional) Unlabeled test images
    train_solution_bounding_boxes.csv  # Original annotations for all images
  ```
- **Split:** We carve out an 80/20 training-validation split from the provided annotations.

## Requirements
- Python 3.7+
- PyTorch
- TorchVision
- pandas
- scikit-learn
- Pillow
- OpenCV-Python
- matplotlib
- pycocotools
- nbformat
- kagglehub (optional, for Kaggle dataset download)

Install via:
```bash
pip install torch torchvision pandas scikit-learn pillow opencv-python matplotlib pycocotools nbformat kagglehub
```

## Setup
1. **Clone the repo** (or download files).
2. **Create a virtual environment**:
   ```bash
   python -m venv env
   source env/bin/activate      # macOS/Linux
   env\Scripts\activate.ps1   # Windows PowerShell (may need `Set-ExecutionPolicy Bypass`)
   ```
3. **Install dependencies** (see Requirements).

## Notebook (`car_detection_model.ipynb`)
The notebook consists of the following sections:
1. **Dataset Preparation**  
   - Download and extract Kaggle dataset via `kagglehub` or manually.  
   - Load `train_solution_bounding_boxes.csv` and split into `train_annots.csv` and `val_annots.csv` (80/20 split).
2. **Custom Dataset Class**  
   - `CarDetectionDataset` parses images and CSV annotations into tensors and target dicts.
3. **DataLoaders**  
   - Create `DataLoader` instances for training and validation.
4. **Model Instantiation**  
   - Load `fasterrcnn_resnet50_fpn`, replace classifier head for 2 classes (`background`, `car`).
5. **Training & Validation Loop**  
   - Train for a configurable number of epochs, logging both train and validation loss.
6. **Inference & Visualization**  
   - Sample detection on validation images and plot bounding boxes.
7. **Discussion**  
   - Analyze loss curves, qualitative performance, limitations, and next steps.

## Running the Notebook
1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
2. Select the `car-env` kernel (if you installed an IPython kernel).
3. Execute cells sequentially.
4. Inspect training/validation losses and sample detections.

## Evaluation (Optional)
To compute COCO-style mAP:
1. Convert `val_annots.csv` to COCO JSON (`val_annotations_coco.json`) via provided script.
2. Run evaluation:
   ```python
   from pycocotools.coco import COCO
   from pycocotools.cocoeval import COCOeval

   coco_gt = COCO("val_annotations_coco.json")
   coco_dt = coco_gt.loadRes("predictions.json")
   coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
   coco_eval.evaluate()
   coco_eval.accumulate()
   coco_eval.summarize()
   ```

## Future Work
- Add richer data augmentations (flips, color jitter, rotations).  
- Hyperparameter tuning (LR, number of epochs, backbone layers).  
- Error analysis on edge cases (occlusions, small objects, nighttime).  
- Deploy model as a REST API or integrate into real-time video pipelines.
