"""
Based on your professor's meeting notes and the project context, here are research ideas with CVPR-level framing:

Idea 1: Compression-Aware Fine-Grained Recognition (Strongest fit)
Core Problem: CCTV footage is H.264-compressed before reaching your model. H.264 uses DCT block compression that destroys high-frequency detail — exactly the fine-grained texture needed to distinguish K/Q/J from 8/9/10.

Novel Contribution:

First systematic study of video compression degradation on fine-grained recognition (distinct from weather/noise degradation which is well-studied)
Propose a Compression-Augmented Training (CAT) framework: during training, simulate H.264/JPEG artifacts at varying quality levels (QP 18–51) so the model learns compression-invariant features
Compare 3 strategies: (a) train on clean → test on compressed, (b) train on compressed → test on compressed, (c) frequency-domain augmentation targeting DCT block artifacts
Experiments:

Sweep compression quality vs. top-1 accuracy on 52-class card dataset
Test across MobileNet, EfficientNet, ViT — show all degrade differently
Super-resolution (ESRGAN/Real-ESRGAN) as preprocessing baseline
Why CVPR: Practical, underexplored, generalizable beyond cards (any CCTV-based recognition: license plates, faces, products)

Idea 2: Occlusion-Robust Recognition via Hand-Object Disentanglement
Core Problem: Dealer hands occlude cards, causing MobileNet to classify hand texture instead of card content.

Novel Contribution:

Hand-Guided Attention Masking: Use the YOLO hand detection bounding box to generate a soft attention mask that suppresses hand regions before feeding crops into the classifier
Train a lightweight occlusion degree estimator (what % of card is covered) — only trust classification when occlusion < threshold, else defer to temporal accumulation
Construct an Occluded Card Recognition Benchmark from your video (first of its kind)
Why CVPR: Connects hand-object interaction (active CVPR topic) with fine-grained recognition under real-world occlusion

Idea 3: State-Constrained Temporal Evidence Accumulation
Core Problem: Single-frame card classification is noisy; the same card is visible across many frames.

Novel Contribution:

Bayesian temporal fusion: accumulate softmax probability vectors across frames using a Kalman-like filter, weighted by occlusion score and detection confidence
Rule-constrained inference: baccarat rules constrain valid states (exactly 4–6 cards, specific player/banker counts). Use this as a structured prior to reject physically impossible predictions
Show that temporal accumulation + rule constraints eliminates most K/Q/J misclassification without retraining
Why CVPR: Temporal reasoning + structured domain knowledge for recognition — connects to neurosymbolic AI, a hot topic

Recommended Direction for Thesis
Primary: Idea 1 (compression degradation analysis + solution) — directly motivated by your data, professor explicitly said "I haven't found any relevant papers that fit our specific situation"

Secondary (adds application chapter): Idea 3 (temporal accumulation) — implementable on top of your current system, directly fixes the K/Q misclassification problem you demonstrated

Combined paper title:

"Compression-Robust Fine-Grained Card Recognition in Surveillance Video via Frequency-Aware Training and Temporal Evidence Accumulation"

Immediate Action Items
Run this experiment this week: Take your clean card crops → compress to JPEG Q=90, 70, 50, 30, 10 → measure per-class accuracy drop. This single experiment IS a paper contribution.
Check per-class confusion matrix — the professor explicitly said K/Q/J confusion is a research point, not just a bug.
Add card back as class 73 — professor flagged this as a correctness issue

___________________________________________________________
So he is suggesting this as your first core academic contribution:

Study how compression/transmission quality degradation affects detection/classification performance.

That is much stronger than just saying “we built a card detector.”
________________________________________________________________







"""