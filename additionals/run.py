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

Inside main.py, it runs a pipeline roughly like:
    table alignment / homography-ish
    optional foreground extraction
    YOLO detection + SORT tracking
    MobileNet card classification
    baccarat logic + overlays + optional recording

The pipeline (mental model):
Video frame → table alignment / masking → YOLO detect → SORT track → crop ROIs (cards/hands) → card classification → baccarat logic + overlay + record





"""