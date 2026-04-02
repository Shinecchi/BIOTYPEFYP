import sys
import os
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.capture.keystroke_logger import KeystrokeLogger
from src.segmentation.sliding_window import SlidingWindowSegmenter
from src.preprocessing.feature_extraction import FeatureExtractor
from src.preprocessing.interpolation import LinearInterpolator
from src.preprocessing.gafmat import GAFMATTransformer
from src.model.siamese_network import BioTypeSiameseModel
from src.training.triplet_trainer import TripletModelBuilder

def generate_synthetic_imposter_features(num_samples: int, target_length: int = 10) -> list:
    """Generates fake (10, 2) feature matrices simulating a different user."""
    synthetic_features = []
    for _ in range(num_samples):
        dwell = np.random.normal(0.15, 0.02, target_length)
        flight = np.random.normal(0.25, 0.05, target_length)
        matrix = np.column_stack((dwell, flight))
        synthetic_features.append(matrix)
    return synthetic_features

def main():
    print("--- BioType Synthetic Training Loop Test ---")
    
    # 1. Initialize Pipeline
    logger = KeystrokeLogger()
    segmenter = SlidingWindowSegmenter(window_size_events=20, step_size_events=10)
    extractor = FeatureExtractor()
    interpolator = LinearInterpolator(target_length=10)
    gafmat = GAFMATTransformer(image_size=10)

    # 2. Capture Genuine Data
    print("\n[Action] Type a very long sentence to generate data (at least 15-20 words).")
    print("Example: 'the quick brown fox jumps over the lazy dog repeatedly until it gets tired'")
    logger.start()
    events = logger.get_events()
    
    windows = segmenter.segment(events)
    if len(windows) < 3:
        print("[Error] Not enough windows to create Anchor/Positive pairs. Type more.")
        return

    # 3. Process Genuine Data into GAFMAT
    genuine_gafmat = []
    for w in windows:
        raw_feat = extractor.extract_features(w)
        fixed_feat = interpolator.process(raw_feat)
        gaf_img = gafmat.transform(fixed_feat)
        genuine_gafmat.append(gaf_img)
    
    genuine_gafmat = np.array(genuine_gafmat)

    # 4. Create Triplets (Anchor, Positive, Negative)
    num_triplets = len(genuine_gafmat) - 1
    anchors = genuine_gafmat[:num_triplets]
    positives = genuine_gafmat[1:]
    
    print(f"\n[Status] Generating {num_triplets} synthetic imposter samples...")
    fake_features = generate_synthetic_imposter_features(num_triplets)
    negatives = np.array([gafmat.transform(f) for f in fake_features])

    print(f"[Data Ready] Anchors: {anchors.shape}, Positives: {positives.shape}, Negatives: {negatives.shape}")

    # 5. Build and Compile the Model
    print("\n[Status] Compiling Siamese Triplet Network...")
    siamese_base = BioTypeSiameseModel(input_shape=(10, 10, 2))
    trainer = TripletModelBuilder(siamese_base.get_base_network())
    model = trainer.build_training_model()

    # 6. Train the Model
    print("\n[Status] Beginning Model Training (5 Epochs)...")
    dummy_y = np.zeros((num_triplets, 1))
    
    history = model.fit(
        x=[anchors, positives, negatives],
        y=dummy_y,
        epochs=5,
        batch_size=4
    )

    # 7. SAVE THE TRAINED WEIGHTS (New Addition)
    # Saves to the root BIOTYPE folder
    weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../BIOTYPE/data/biotype_trained_weights.weights.h5'))
    siamese_base.get_model().save_weights(weights_path)

    print(f"\n[Success] Training completed! Weights saved to: {weights_path}")

if __name__ == "__main__":
    main()