# Automated Livestock Behavior Detection Pipeline

A robust computer vision pipeline for automated detection and classification of animal behaviors in video footage, utilizing foundation models and zero-shot detection capabilities.

**Paper**: [ArXiv Link](https://arxiv.org/pdf/2509.12047)

## Pipeline Overview

![Pipeline Architecture](Images/Fig%201.jpg)

This pipeline implements a four-stage approach for automated behavior detection:
1. **Frame Extraction**: Decode video frames with timestamp preservation
2. **Object Detection & Tracking**: Zero-shot detection using OWLv2 and segmentation/tracking with SAM2
3. **Feature Extraction**: Extract embeddings using DINOv2 foundation model
4. **Behavior Classification**: Multi-class behavior classification using MLP classifier

## Data Flow

```
Video Input
    │
    ├─→ [1. Frame Decoder] ──→ Decoded frames + timestamps
    │                              │
    │                              ├─→ [2a. OWLv2 Detection] ──→ Bounding boxes
    │                              │
    │                              └─→ [2b. SAM2 Segmentation] ──→ Tracked segments
    │                                       │
    │                                       └─→ [3. Frame Cropping] ──→ Cropped animal frames
    │                                                │
    │                                                └─→ [4. DINOv2 Embedding] ──→ Feature vectors
    │                                                         │
    └─→ [Ground Truth Labels] ────────────────────────────────┴─→ [5. Metadata Generation]
                                                                        │
                                                                        └─→ [6. MLP Classifier] ──→ Behavior Predictions
```

## Benchmark Results

### Edinburgh Pig Behavior Dataset

**Object Detection (OWLv2)**
| Metric | Value |
|--------|-------|
| Average Precision (AP) | 89.28% |
| Precision | 80.19% |
| Recall | 88.05% |
| F1 Score | 83.94% |
| Average IoU | 0.747 |

**Segmentation & Tracking (SAM2)**
| Sequence | Idf1 | Recall | Precision | Mota | Num Switches |
|----------|------|--------|-----------|------|--------------|
| 2019_11_05_000002 | 91.0% | 91.0% | 91.0% | 82.0% | 0 |
| 2019_11_11_000028 | 87.4% | 87.4% | 87.4% | 74.8% | 2 |
| 2019_11_11_000036 | 91.6% | 91.6% | 91.6% | 83.2% | 0 |
| 2019_11_22_000010 | 86.4% | 86.4% | 86.4% | 72.8% | 0 |
| 2019_11_28_000113 | 98.3% | 98.3% | 98.3% | 96.6% | 0 |
| 2019_12_02_000005 | 94.5% | 94.5% | 94.5% | 89.1% | 0 |
| 2019_12_02_000208 | 98.1% | 98.1% | 98.1% | 96.2% | 0 |
| 2019_12_10_000060 | 99.9% | 99.9% | 99.9% | 99.8% | 0 |
| 2019_12_10_000078 | 92.8% | 92.8% | 92.8% | 85.6% | 2 |
| **Average** | **93.33%** | **93.33%** | **93.33%** | **86.67%** | **0.44** |

**Behavior Classification (MLP + DINOv2)**
| Behavior | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Standing | 0.892 | 0.762 | 0.821 | 475 |
| Lying | 0.816 | 0.962 | 0.883 | 478 |
| Eating | 0.965 | 0.996 | 0.980 | 821 |
| Drinking | 0.860 | 0.896 | 0.878 | 96 |
| Sitting | 0.662 | 0.878 | 0.754 | 49 |
| Sleeping | 0.992 | 0.937 | 0.964 | 2,289 |
| Running | 0.473 | 0.643 | 0.546 | 14 |
| Playing with toy | 0.900 | 0.947 | 0.923 | 19 |
| Nose-to-nose | 0.492 | 0.938 | 0.645 | 64 |
| **Weighted Average** | **0.940** | **0.929** | **0.932** | **4,305** |

### CBVD-5 Dataset (Images)

| Model | Feature Type | Accuracy | Precision | Recall | F1-Score |
|-------|--------------|----------|-----------|--------|----------|
| MLP | DINOv2 | 98.3% | 0.982 | 0.982 | 0.982 |
| MLP | CLIP | 98.2% | 0.981 | 0.983 | 0.982 |

### Play Behavior Dataset (Dairy Calves)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Active Playing | 0.96 | 0.99 | 0.98 | 2,536 |
| Non-Active Playing | 0.98 | 0.96 | 0.97 | 2,536 |
| Not Playing | 0.99 | 0.98 | 0.98 | 2,537 |
| **Overall Accuracy** | **0.976** | | | **7,609** |

## Pipeline Components

### 1. Frame Extraction
- **Notebook**: `1_Video_Decoding.ipynb`
- Extracts frames from video with timestamp preservation
- Outputs: Decoded frames organized by video segments

### 2a. Object Detection (Optional)
- **Notebook**: `2_OWLV.ipynb` or `2_YOLO.ipynb`
- Zero-shot detection using OWLv2 or traditional YOLO
- Outputs: Bounding box annotations

### 3. Segmentation & Tracking
- **Notebooks**: 
  - `3_Samurai_Usage_cleaned.ipynb` - SAM2 segmentation and tracking
  - `3.5_Samurai_Output_Verification.ipynb` - Quality verification
- Outputs: Tracked object masks across frames

### 4. Frame Cropping
- **Notebook**: `4_Cropping_the_frames_using_annotations.ipynb`
- Crops individual animal frames based on segmentation masks
- Outputs: Individual cropped frames per tracked object

### 5. Feature Extraction
- **Notebook**: `5_Embedding_Extraction_DinoV2.ipynb`
- Extracts DINOv2 embeddings from cropped frames
- Parallel processing for efficient extraction
- Outputs: `.pt` files containing feature vectors

### 6. OCR + Final Metadata Generation
- **Notebook**: `7_final_metadata_for_classification.ipynb` and `6_OCR_Metadata.ipynb`
- Merges all metadata sources (OCR, ground truth, embeddings)
- Outputs: Comprehensive CSV with frame paths, labels, and embedding paths

### 7. Classification
- **Notebook**: `8_MLP_Classifer.ipynb`
- Trains MLP classifier on extracted embeddings
- Includes early stopping and evaluation metrics
- Supports multi-class behavior classification

## Installation

```bash
# Clone the repository
git clone https://github.com/Sonam525/livestock-behavior-detection.git
cd livestock-behavior-detection

# Install dependencies
pip install torch torchvision transformers ultralytics opencv-python pyyaml
pip install pandas numpy scikit-learn
pip install easyocr  # For OCR-based timestamp extraction

# For SAM2 segmentation
pip install segment-anything-2
```

## Usage

### Quick Start

1. **Prepare your video data**:
   - Place videos in a designated input folder
   - Ensure ground truth labels are available (if training)

2. **Run the pipeline sequentially**:

```python
# 1. Extract frames
# Run Frames_Decoding.ipynb with your video paths

# 2. Detect and track objects
# Run OWLV.ipynb or YOLO.ipynb for detection
# Run Samurai_Usage_cleaned.ipynb for segmentation/tracking

# 3. Crop frames
# Run Cropping_the_frames_using_annotations.ipynb

# 4. Extract embeddings
# Run Embedding_Extraction_DinoV2.ipynb

# 5. Generate metadata
# Run Metadata.ipynb and Generate_the_Final_Metadata.ipynb

# 6. Train classifier
# Run MLP_Classifer.ipynb
```

### Configuration

Key parameters to adjust in notebooks:

**Frame Extraction:**
- `fps`: Frame extraction rate
- `output_dir`: Destination for decoded frames

**Object Detection:**
- `confidence_threshold`: Detection confidence (default: 0.3)
- `text_prompts`: Object classes for OWLv2

**Embedding Extraction:**
- `max_workers`: Parallel processing workers (default: 10)
- `MAX_INDEX`: Maximum frame index to process

**Classification:**
- `batch_size`: Training batch size (default: 64)
- `hidden_dims`: MLP architecture (default: [512, 256])
- `dropout`: Regularization (default: 0.5)
- `learning_rate`: Optimizer learning rate (default: 1e-3)

## Project Structure

```
.
├── 1_Video_Decoding.ipynb                      # Step 1: Video frame extraction
├── 2_OWLV.ipynb                                 # Step 2a: OWLv2 object detection
├── 2_YOLO.ipynb                                 # Step 2a: YOLO detection (alternative)
├── 3_Samurai_Usage_cleaned.ipynb                # Step 3: SAM2 segmentation
├── 3.5_Samurai_Output_Verification.ipynb          # Step 3b: Verification
├── 4_Cropping_the_frames_using_annotations.ipynb # Step 4: Frame cropping
├── 5_Embedding_Extraction_DinoV2.ipynb          # Step 5: Feature extraction
├── 7_final_metadata_for_classification.ipynb & 6_OCR_Metadata.ipynb    # Step 6: Metadata merging
├── 8_MLP_Classifer.ipynb                        # Step 7: Behavior classification
├── Images/
│   └── Fig 1.jpg                              # Pipeline visualization
└── README.md
```

## Key Features

- **Zero-shot Detection**: Uses OWLv2 for detection without fine-tuning on domain-specific data
- **Foundation Model Features**: Leverages DINOv2 for robust visual representations
- **Scalable Processing**: Parallel processing for efficient embedding extraction
- **Comprehensive Tracking**: SAM2-based segmentation maintains identity across frames
- **Multi-dataset Validation**: Benchmarked on pig behavior, cattle behavior, and play behavior datasets

## Model Performance Summary

| Dataset | Task | Best Model | Accuracy/Metric |
|---------|------|------------|-----------------|
| Edinburgh Pigs | Detection | OWLv2 | 89.28% AP |
| Edinburgh Pigs | Tracking | SAM2 | 93.33% Idf1 |
| Edinburgh Pigs | Classification | MLP+DINOv2 | 93.2% F1 |
| CBVD-5 | Classification | MLP+DINOv2 | 98.3% Accuracy |
| Play Behavior | Classification | MLP+DINOv2 | 97.6% Accuracy |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for faster processing)
- Databricks environment (notebooks optimized for Databricks)
  - Can be adapted for local execution with minor modifications

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{yang2025computervisionpipelineindividuallevel,
      title={A Computer Vision Pipeline for Individual-Level Behavior Analysis: Benchmarking on the Edinburgh Pig Dataset}, 
      author={Haiyu Yang and Enhong Liu and Jennifer Sun and Sumit Sharma and Meike van Leerdam and Sebastien Franceschini and Puchun Niu and Miel Hostens},
      year={2025},
      eprint={2509.12047},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.12047}, 
}
```

## License

MIT License

## Acknowledgments

- OWLv2 by Google Research
- DINOv2 by Meta AI Research
- SAM2 (Segment Anything 2) by Meta AI Research
- YOLOv8 by Ultralytics

## Contact

For questions or issues, please open an issue on GitHub.

## Future Work

- Real-time processing pipeline
- Integration with edge devices
- Extended behavior categories
- Multi-species adaptation
