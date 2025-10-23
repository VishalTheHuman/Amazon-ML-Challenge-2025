# Amazon ML Challenge 2025: Smart Product Pricing Challenge

**Team Name:** CTRL + ALT + DEV

**Team Members:** 

- Vishal S
- Krishna Leela Amritha Nandini
- Suriya KP
- SP Saran Dharshan

**Submission Date:** 13 October 2025

---

## Executive Summary

A comprehensive analysis of multiple approaches attempted for the Amazon ML Challenge 2025, focusing on predicting grocery product prices from textual descriptions and images. The best performing approach achieved a **SMAPE score of 40.777%** on 25k Entires which are set as public dataset using a sophisticated multi-modal fusion architecture combining CLIP and DistilBERT models with contrastive learning. This report details the winning approach and summarizes all other attempted methodologies.

**NOTE :**

The final submission is the average of the best 2 models.

---

## 1. Best Performing Approach: Multi-Modal Fusion with Contrastive Learning

**SMAPE Score: 40.777%**

**Date: 13 Oct 25, 08:59 PM IST**

### 1.1 Architecture Overview

The winning solution employs a **dual-backbone fusion architecture** that combines:

-**CLIP (Vision-Language Model)**: `openai/clip-vit-large-patch14` for image-text understanding

-**DistilBERT**: `distilbert-base-uncased-finetuned-sst-2-english` for enhanced text processing

-**Contrastive Learning**: Multiple InfoNCE losses for representation learning

-**Fusion Head**: Multi-layer perceptron combining all modalities

### 1.2 Model Architecture Details

```

Input: Product Text + Product Image

    ↓

┌─────────────────┬─────────────────┐

│   CLIP Branch   │ DistilBERT Branch│

│                 │                 │

│ Text → CLIP     │ Text → Distil   │

│ Image → CLIP    │ (2 augmented    │

│                 │  views)         │

└─────────────────┴─────────────────┘

    ↓

┌─────────────────────────────────────┐

│        Feature Fusion               │

│ [norm(img_clip), norm(txt_clip),    │

│  norm(distil_proj)] → MLP → Price   │

└─────────────────────────────────────┘

```

### 1.3 Key Technical Innovations

#### Multi-Modal Feature Extraction

-**CLIP Text Features**: 768-dimensional embeddings from product descriptions

-**CLIP Image Features**: 768-dimensional visual representations from product images

-**DistilBERT Features**: 256-dimensional projected embeddings from text (two augmented views)

#### Advanced Contrastive Learning

1.**CLIP Image-Text Contrastive Loss**: Aligns visual and textual representations

- InfoNCE loss between CLIP image and text features
- Temperature parameter: τ = 0.07
- Weight: α_clip = 0.20

2.**DistilBERT SimCSE-style Contrastive Loss**: Improves text representations

- InfoNCE loss between two augmented text views
- Word masking probability: 6%
- Weight: α_txt = 0.10

#### Robust Training Strategy

-**Multi-Loss Optimization**: Combines regression (Huber loss) with contrastive losses

-**Missing Image Handling**: Zero-padding policy for missing images

-**Log2 Price Transformation**: Handles price distribution skewness

-**Gradient Accumulation**: Batch size 16 with gradient accumulation

-**Mixed Precision Training**: FP16 for memory efficiency

### 1.4 Training Configuration

```python

# Model Parameters

- CLIP Model: openai/clip-vit-large-patch14

- DistilBERT: distilbert-base-uncased-finetuned-sst-2-english

- Total Parameters: 501,196,546 trainable parameters

- Fusion Head: 2x hidden dimension expansion


# Training Hyperparameters

- Learning Rate: 2e-5

- Batch Size: 16

- Epochs: 15

- Warmup Ratio: 6%

- Weight Decay: 0.01

- Max Gradient Norm: 1.0


# Loss Weights

- Regression Loss: 1.0(Huber loss, δ=1.0)

- CLIP Contrastive: 0.20

- DistilBERT Contrastive: 0.10

```

### 1.5 Data Processing Pipeline

1.**Text Preprocessing**:

- CLIP: Max length 64 tokens
- DistilBERT: Max length 192 tokens with word masking

2.**Image Processing**:

- Resize to 224x224 pixels
- RGB conversion
- CLIP vision encoder processing

3.**Missing Data Handling**:

- Zero-padding for missing images
- No data dropping to preserve sample size

### 1.6 Why This Approach Succeeded

1.**Multi-Modal Understanding**: Leverages both visual and textual information effectively

2.**Contrastive Learning**: Improves representation quality through self-supervised learning

3.**Robust Architecture**: Handles missing images gracefully without performance degradation

4.**Optimal Hyperparameters**: Carefully tuned loss weights and training parameters

5.**Advanced Fusion**: Sophisticated feature combination strategy

---

## 2. Other Approaches Summary

### 2.1 CLIP-Based Approaches (SMAPE: 41.541% - 42.234%)

**Approaches:**

-**13 Oct 25, 08:27 AM IST (41.541%)**: `openai/clip-vit-large-patch14-336` with 10 epochs

-**12 Oct 25, 11:28 PM IST (42.234%)**: `openai/clip-vit-large-patch14` with 5 epochs

**Key Features:**

- Single CLIP model for vision-language understanding
- Contrastive learning between image and text features
- Huber loss for robust regression
- Log2 price transformation

**Limitations:** Single modality fusion, less sophisticated text processing compared to winning approach.

### 2.2 BERT-Based Approaches (SMAPE: 45.429% - 48.624%)

**Approaches:**

-**12 Oct 25, 09:02 PM IST (48.624%)**: `google-bert/bert-large-uncased` with contrastive learning

-**12 Oct 25, 08:20 PM IST (45.429%)**: `distilbert-base-uncased-finetuned-sst-2-english`

**Key Features:**

- Text-only processing with BERT variants
- Contrastive learning with text augmentation
- Two-stage training (validation + full data)
- SimCSE-style text representation learning

**Limitations:** No visual information utilization, limited to textual features only.

### 2.3 Embedding-Based Approaches (SMAPE: 50.702% - 57.402%)

**Approaches:**

-**12 Oct 25, 12:00 AM IST (50.702%)**: `intfloat/e5-base-v2` embeddings

-**11 Oct 25, 11:52 PM IST (50.935%)**: `intfloat/e5-base-v2` embeddings

-**11 Oct 25, 05:06 PM IST (57.402%)**: `Qwen/Qwen3-Embedding-0.6B` with LightGBM

**Key Features:**

- Pre-trained embedding models for text representation
- Traditional ML models (LightGBM) for regression
- Feature engineering from embeddings
- Ensemble methods

**Limitations:** Static embeddings, no end-to-end learning, limited to text-only processing.

### 2.4 Traditional ML with Feature Engineering (SMAPE: 50.935% - 51.491%)

**Approaches:**

-**11 Oct 25, 11:29 PM IST (51.491%)**: LightGBM with comprehensive feature engineering

-**11 Oct 25, 11:52 PM IST (50.935%)**: E5 embeddings with LightGBM

**Key Features:**

- Extensive feature engineering from product descriptions
- Brand detection, size extraction, keyword analysis
- TF-IDF and LDA features
- Tree-based ensemble models

**Limitations:** Manual feature engineering, no visual information, limited representation learning.

---

## 3. Performance Analysis

### 3.1 SMAPE Score Ranking - Personal

| Rank | Date | SMAPE | Approach |

|------|------|-------|----------|

| 1 | 13 Oct 25, 08:59 PM | **40.777%** | CLIP + DistilBERT Fusion |

| 2 | 13 Oct 25, 08:27 AM | 41.541% | CLIP Large (10 epochs) |

| 3 | 13 Oct 25, 05:50 PM | 42.006% | CLIP + E5 Embeddings |

| 4 | 12 Oct 25, 11:28 PM | 42.234% | CLIP Large (5 epochs) |

| 5 | 12 Oct 25, 09:02 PM | 48.624% | BERT Large |

| 6 | 12 Oct 25, 08:20 PM | 45.429% | DistilBERT SST-2 |

| 7 | 12 Oct 25, 04:33 PM | 45.852% | DistilBERT Base |

| 8 | 12 Oct 25, 12:00 AM | 50.702% | E5 Base v2 |

| 9 | 11 Oct 25, 11:52 PM | 50.935% | E5 Base v2 |

| 10 | 11 Oct 25, 11:29 PM | 51.491% | LightGBM + Features |

| 11 | 11 Oct 25, 05:06 PM | 57.402% | Qwen Embeddings |

### 3.2 Key Insights

1.**Multi-Modal Approaches Dominate**: Top 4 approaches all use visual information

2.**Contrastive Learning Effectiveness**: All top performers use some form of contrastive learning

3.**Model Size Matters**: Larger models (CLIP Large, BERT Large) perform better

4.**Fusion Architecture Advantage**: The winning approach's dual-backbone fusion provides significant improvement

5.**Training Duration Impact**: Longer training (15 epochs vs 5-10) shows improvement

---

## 4. Technical Lessons Learned

### 4.1 What Worked

1.**Multi-Modal Fusion**: Combining visual and textual information significantly improves performance

2.**Contrastive Learning**: InfoNCE losses help learn better representations

3.**Robust Loss Functions**: Huber loss handles price outliers effectively

4.**Careful Hyperparameter Tuning**: Loss weights and temperature parameters are crucial

5.**Missing Data Handling**: Zero-padding strategy preserves sample size without performance loss

### 4.2 What Didn't Work

1.**Text-Only Approaches**: Limited performance without visual information

2.**Static Embeddings**: Pre-trained embeddings without fine-tuning underperform

3.**Simple Feature Engineering**: Manual features cannot capture complex product relationships

4.**Single Modality**: CLIP alone or BERT alone cannot match multi-modal performance

## 5. Conclusion

The winning approach demonstrates the power of **sophisticated multi-modal fusion** with **contrastive learning** for price prediction tasks. The combination of CLIP and DistilBERT with carefully tuned loss functions and training strategies achieved a **40.777% SMAPE score**, significantly outperforming all other approaches.

**Key Success Factors:**

1. Multi-modal understanding of products
2. Advanced contrastive learning techniques
3. Robust architecture design
4. Careful hyperparameter optimization
5. Effective handling of missing data

This approach provides a strong foundation for production deployment and can be further improved with additional data augmentation, ensemble methods, and more sophisticated fusion strategies.

## Appendix

### A. Model Comparison Summary

| Approach Type | Best SMAPE | Key Components | Limitations |

|---------------|------------|----------------|-------------|

| Multi-Modal Fusion | **40.777%** | CLIP + DistilBERT + Contrastive | High computational cost |

| CLIP-Only | 41.541% | Single CLIP model | Limited text processing |

| BERT-Only | 45.429% | Text-only processing | No visual information |

| Embedding + ML | 50.702% | Static embeddings + LightGBM | No end-to-end learning |

| Feature Engineering | 51.491% | Manual features + LightGBM | Limited representation |

### B. Technical Specifications

**Winning Model:**

- Architecture: Dual-backbone fusion (CLIP + DistilBERT)
- Parameters: 501,196,546 trainable
- Training: 15 epochs, 2e-5 learning rate
- Loss: Huber + 2x InfoNCE contrastive losses

**Performance Metrics:**

- SMAPE: 40.777%
- Training Time: ~5 hours
