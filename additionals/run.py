"""
python main.py gkl_table1_layout.jpg gkl_mask1.avi
python augment_dataset.py --target-size 224 224

# Step 1 — compute mean/std once
python mobilenet_card_classifier.py stats dataset_gkl_cards/augmented

# Step 2 — train
python mobilenet_card_classifier.py train dataset_gkl_cards/augmented

# Step 3 — evaluate
python mobilenet_card_classifier.py test  dataset_gkl_cards/augmented/val







# Show on screen only — no file saved  (original behaviour)
python main.py gkl_table1_layout.jpg gkl_mask1.avi

# Save to default filename  baccarat_annotated.mp4  AND show on screen
python main.py gkl_table1_layout.jpg gkl_mask1.avi --save

# Save to a custom filename
python main.py gkl_table1_layout.jpg gkl_mask1.avi --save --output my_run.mp4

# Save without showing on screen (headless server / SSH)
python main.py gkl_table1_layout.jpg gkl_mask1.avi --save --no-viz

python compress_video.py --video gkl_mask1.avi







python compress_val_dataset.py
Val only (default CRFs: 0 18 23 28 35 42 47 51):


python compress_val_dataset.py
Val + Train at CRF 28 and 38:
python compress_val_dataset.py --train

Train only at CRF 28 and 38 (skip val):
python compress_val_dataset.py --crf none --train

Actually that won't work cleanly — if you want train-only, just run:
python compress_val_dataset.py --crf 28 38 --train --crf-train 28 38


Done. Here's how to run the three training experiments:

Clean training (existing):


python mobilenet_card_classifier.py train
# checkpoints → working/
CRF 28 compressed training:


python mobilenet_card_classifier.py train --crf-train 28
# checkpoints → working/crf28/
CRF 38 compressed training:


python mobilenet_card_classifier.py train --crf-train 38
# checkpoints → working/crf38/






Step 1 — Delete the old augmented data and regenerate:
rm -rf dataset_gkl_cards/augmented
python augment_dataset.py
Step 2 — Recompute dataset stats (mean/std changed because augmented images changed):


rm working/dataset_stats.json
python mobilenet_card_classifier.py stats
Step 3 — Retrain from scratch:


python mobilenet_card_classifier.py train
This time you should expect the accuracy to be lower — somewhere around 85–95% depending on the class. That lower number is the honest number. If it's still 99%+, the remaining issue is that the data all comes from one video session (same deck, same conditions).

Step 4 — Then train CRF variants:


python compress_val_dataset.py --train
python mobilenet_card_classifier.py train --crf-train 28
python mobilenet_card_classifier.py train --crf-train 38


rm working/best.pt working/best_ep*.pt working/history.pickle
python mobilenet_card_classifier.py train





Good — already removed cleanly. Now run the complete pipeline from scratch:


# Step 1 — delete old augmented data and regenerate with all fixes
rm -rf dataset_gkl_cards/augmented
python augment_dataset.py
Watch the output — you will now see val_imgs= showing the real unique count per class:


[  9/ 53]  9 clubs        src=  225  runs=  2  train_runs=1  val_runs=1  val_imgs=16
[ 10/ 53]  3 clubs        src=  245  runs=  4  train_runs=3  val_runs=1  val_imgs=7

# Step 2 — recompute stats from the new augmented data
rm working/dataset_stats.json
python mobilenet_card_classifier.py stats

# Step 3 — train from scratch
rm -f working/best.pt working/best_ep*.pt working/history.pickle
python mobilenet_card_classifier.py train --epochs 20






# 1. Regenerate augmented data with the per-frame 80/20 temporal split
rm -rf dataset_gkl_cards/augmented
python augment_dataset.py

# 2. Recompute mean/std (augmented images changed)
rm -f working/dataset_stats.json
python mobilenet_card_classifier.py stats

# 3. Train clean model from scratch
rm -f working/best.pt working/best_ep*.pt working/history.pickle
python mobilenet_card_classifier.py train --epochs 20

# 4. Generate CRF-compressed val sets (CRFs 0 18 23 28 35 42 47 51)
python compress_val_dataset.py

# 5. (Optional) Generate CRF-compressed training sets for CRF-28 / CRF-38 experiments
python compress_val_dataset.py --train

# 6. Train CRF variants
python mobilenet_card_classifier.py train --crf-train 28 --epochs 20
python mobilenet_card_classifier.py train --crf-train 38 

###########################################
 Step 1: Compress val (CRFs 0 18 23 28 35 42 47 51) + train (CRFs 28 38) in one shot
python compress_val_dataset.py --train

# Step 2: Train on CRF-28 compressed training data
python mobilenet_card_classifier.py train --crf-train 28 --epochs 20

# Step 3: Train on CRF-38 compressed training data
python mobilenet_card_classifier.py train --crf-train 38 --epochs 20

# Step 4: Plot all three experiments together
python plot_training.py --compare






Done. Run the evaluation with:
python compression_eval.py
This uses working/best.pt by default and evaluates against clean val + CRFs 0, 18, 23, 28, 35, 42, 47, 51. Output goes to working/compression_eval/

# Train on CRF-28 compressed training data
python mobilenet_card_classifier.py train --crf-train 28 --epochs 20

# Train on CRF-38 compressed training data
python mobilenet_card_classifier.py train --crf-train 38 --epochs 20

# Compare all three models on the CRF degradation curve
python compression_eval.py --checkpoint working/crf28/best.pt \
    --output working/compression_eval_crf28

python compression_eval.py --checkpoint working/crf38/best.pt \
    --output working/compression_eval_crf38





Inside main.py, it runs a pipeline roughly like:
    table alignment / homography-ish
    optional foreground extraction
    YOLO detection + SORT tracking
    MobileNet card classification
    baccarat logic + overlays + optional recording

The pipeline (mental model):
Video frame → table alignment / masking → YOLO detect → SORT track → crop ROIs (cards/hands) → card classification → baccarat logic + overlay + record





"""