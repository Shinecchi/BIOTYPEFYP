import sys
import os
import matplotlib.pyplot as plt

# Ensure the project root is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.capture.keystroke_logger import KeystrokeLogger
from src.segmentation.sliding_window import SlidingWindowSegmenter
from src.preprocessing.feature_extraction import FeatureExtractor
from src.preprocessing.interpolation import LinearInterpolator
from src.preprocessing.gafmat import GAFMATTransformer

def main():
    print("--- BioType GAFMAT Visualizer ---")
    
    # Initialize components
    logger = KeystrokeLogger()
    segmenter = SlidingWindowSegmenter(window_size_events=20, step_size_events=10)
    extractor = FeatureExtractor()
    interpolator = LinearInterpolator(target_length=10)
    gafmat = GAFMATTransformer(image_size=10)

    # Capture data
    print("\n[Action] Type a natural sentence (e.g., 'behavioral biometrics are cool').")
    logger.start()
    events = logger.get_events()

    if len(events) < 20:
        print("[Error] Not enough data. Please type at least one full word.")
        return

    # Process only the FIRST window for the visualization
    first_window = segmenter.segment(events)[0]
    raw_features = extractor.extract_features(first_window)
    fixed_features = interpolator.process(raw_features)
    
    # This is the (10, 10, 2) image tensor
    gaf_image = gafmat.transform(fixed_features)

    # --- Plotting the Image ---
    # We create a side-by-side plot for Channel 0 (Dwell) and Channel 1 (Flight)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot Dwell Time GAF
    cax1 = axes[0].imshow(gaf_image[:, :, 0], cmap='jet', origin='lower')
    axes[0].set_title("Channel 0: Dwell Time Pattern")
    fig.colorbar(cax1, ax=axes[0])

    # Plot Flight Time GAF
    cax2 = axes[1].imshow(gaf_image[:, :, 1], cmap='jet', origin='lower')
    axes[1].set_title("Channel 1: Flight Time Pattern")
    fig.colorbar(cax2, ax=axes[1])

    plt.suptitle("Your Keystroke Behavioral Signature (10x10 GAFMAT)")
    plt.tight_layout()
    
    print("[Status] Rendering image window... Close the image window to end the script.")
    plt.show()

if __name__ == "__main__":
    main()