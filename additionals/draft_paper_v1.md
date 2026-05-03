# Impact of H.264 Video Compression on Deep Learning-based Playing Card Recognition in AI Casino Table Monitoring

**[Author Name], [Professor Name]**
[Department / University Name]
{email@university.ac.kr}

---

## Abstract

This paper investigates the robustness of a deep learning-based playing card classification system against H.264 video compression, in the context of an automated baccarat table monitoring application. A MobileNetV1-1.25× model is trained on a card dataset collected from a live casino table and evaluated against validation images compressed at eight H.264 Constant Rate Factor (CRF) levels (0–51). Experimental results show that Top-1 accuracy remains stable at approximately 97.5–97.8% across CRF levels 0–42, but degrades significantly at CRF 47 (92.8%) and CRF 51 (78.0%). We further study whether training on compressed data improves robustness at high CRF levels. [Results for CRF-28 and CRF-38 trained models to be added after experiments complete.]

**Keywords:** card recognition, MobileNet, H.264 compression, casino monitoring, baccarat, deep learning

---

## 1. Introduction

Automated monitoring of casino table games is increasingly important for fraud detection, dealer assistance, and game analytics. A complete monitoring pipeline must process video from ceiling-mounted cameras that use real-time H.264 compression — the dominant codec in IP-camera systems. While deep learning models for image classification are well studied in clean-image settings, their robustness under varying levels of codec compression remains underexplored.

In this work, we build an end-to-end baccarat monitoring system that integrates YOLO-based hand detection, SORT multi-object tracking, and MobileNet card classification. We then conduct a systematic study of how H.264 CRF compression levels affect the card recognition accuracy, and whether training the classifier on compressed images can compensate for the quality loss.

---

## 2. System Architecture

The monitoring pipeline consists of three stages (Fig. 1):

1. **Detection**: A fine-tuned YOLOv11-nano model detects baccarat hands (player / banker) and individual card regions in each video frame.
2. **Tracking**: SORT (Simple Online and Realtime Tracking) associates detections across frames, maintaining stable card identities through occlusion.
3. **Classification**: Cropped card regions are classified into 53 categories (52 standard cards + back-face) using a fine-tuned MobileNetV1-1.25× model (timm: `mobilenetv1_125.ra4_e3600_r224_in1k`).

The baccarat game logic layer reads the classified cards and updates player/banker hand totals in real time, overlaying results on the output stream.

---

## 3. Card Classification Model

### 3.1 Dataset

Card images were extracted from a live GKL (Grand Korea Leisure) baccarat table recording. Source images are consecutive video frames of each card lying on the felt table. To prevent temporal data leakage between training and validation, images are grouped into **consecutive-frame runs** (one run = one card position). The last 25% of runs per class are assigned entirely to the validation split; the remaining 75% are used for training. This ensures that train and val sets contain different card positions, preventing the model from memorizing specific frame sequences.

Training images are augmented to 1,200 images per class using rotation (±10°), brightness/contrast jitter, HSV perturbation, Gaussian blur/noise, random perspective, and random crop-and-resize. Horizontal flipping is intentionally disabled to avoid confusing visually symmetric digits (6↔9). The final dataset contains 63,600 training images and 7,932 validation images across 53 classes.

### 3.2 Training

The MobileNetV1-1.25× backbone (pretrained on ImageNet) is fine-tuned with:

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (weight_decay = 1e-4) |
| Learning rate | 3×10⁻⁴ (OneCycleLR, 30% warmup → cosine anneal) |
| Batch size | 64 |
| Epochs | 20 |
| Dropout | 0.3 |
| Label smoothing | 0.1 |

The best checkpoint achieves **98.10% Top-1 validation accuracy** (epoch 18, val_loss = 0.166).

---

## 4. Compression Degradation Study

### 4.1 Method

H.264 compression is simulated by passing each validation image through an ffmpeg libx264 encode–decode cycle at a fixed CRF value (0 = near-lossless, 51 = worst quality). CRF levels tested: **0, 18, 23, 28, 35, 42, 47, 51**.

Three model variants are evaluated:
- **Clean**: trained on uncompressed augmented images (working/best.pt)
- **CRF-28**: trained on images compressed at CRF 28 (in progress)
- **CRF-38**: trained on images compressed at CRF 38 (in progress)

### 4.2 Results

Table 1 shows Top-1 accuracy of the clean-trained model across CRF levels.

**Table 1. Top-1 accuracy of clean-trained model vs. H.264 CRF level (7,932 val images)**

| Condition | CRF | Top-1 Acc (%) | Val Loss | Drop vs. clean |
|---|---|---|---|---|
| No compression (baseline) | — | **98.10** | 0.166 | — |
| H.264 near-lossless | 0 | 98.11 | 0.166 | 0.00 pp |
| Near-lossless | 18 | 98.13 | 0.166 | 0.00 pp |
| Default encoder quality | 23 | 98.13 | 0.167 | 0.00 pp |
| Moderate compression | 28 | 98.15 | 0.169 | 0.00 pp |
| Heavy compression | 35 | 98.13 | 0.172 | 0.00 pp |
| Very heavy compression | 42 | 97.63 | 0.207 | −0.47 pp |
| Severe compression | 47 | 92.36 | 0.456 | **−5.74 pp** |
| Worst quality | 51 | 76.50 | 1.030 | **−21.60 pp** |

**Key finding**: Accuracy is completely stable (≈98.1%) across CRF 0–42 — covering all practical IP-camera deployment scenarios (typically CRF 18–28). A sharp accuracy cliff occurs at CRF 47 (−5.74 pp), with severe degradation at CRF 51 (−21.60 pp). The accuracy plateau at moderate CRF levels is explained by the nature of card features: rank numerals and suit symbols are large, high-contrast structures that survive H.264 block-based compression until extreme quantisation levels destroy their shape.

*[Table 2 — comparison of clean vs. CRF-28 vs. CRF-38 trained models will be added once training experiments complete by this weekend.]*

---

## 5. Conclusion

We present a real-time AI baccarat table monitoring system and conduct a systematic study of H.264 compression impact on deep learning-based card recognition. The MobileNet classifier achieves 98.10% validation accuracy and remains fully robust (≈98.1%) across all H.264 CRF levels up to 42, which encompasses every practical IP-camera deployment setting. Meaningful accuracy degradation begins only at CRF 47 (−5.7 pp) and becomes severe at CRF 51 (−21.6 pp). This finding confirms that the system is suitable for deployment in real casino environments with standard video compression. Ongoing experiments will quantify whether training on compressed images further extends robustness at extreme CRF levels (47–51).

---

## References

[1] A. G. Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," arXiv:1704.04861, 2017.

[2] R. Bewley et al., "Simple Online and Realtime Tracking," in Proc. ICIP, 2016.

[3] G. Jocher et al., "Ultralytics YOLO," 2023. [Online]. Available: https://github.com/ultralytics/ultralytics

[4] R. Wightman, "PyTorch Image Models (timm)," 2019. [Online]. Available: https://github.com/rwightman/pytorch-image-models

[5] [Any related casino/card recognition paper you can find and cite here]

---

*Draft v1 — May 1, 2026. Remaining: CRF-28 and CRF-38 training experiments + comparison table + Jetson Nano deployment results.*
