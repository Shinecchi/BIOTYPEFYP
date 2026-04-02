import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.model.siamese_network import BioTypeSiameseModel

def main():
    print("--- BioType Siamese Neural Network Test ---")
    
    # 1. Initialize the Model
    model_builder = BioTypeSiameseModel(input_shape=(10, 10, 2))
    siamese_net = model_builder.get_model()
    
    # Print the architecture summary
    print("\n[Architecture Summary]")
    model_builder.get_base_network().summary()

    # 2. Create Dummy GAFMAT Images (Simulating Window 1 and Window 2)
    # Shape: (batch_size, height, width, channels)
    print("\n[Status] Generating mock (10, 10, 2) GAFMAT tensors...")
    current_typing_window = np.random.rand(1, 10, 10, 2)
    reference_typing_profile = np.random.rand(1, 10, 10, 2)

    # 3. Run Inference (Predict Distance)
    print("[Status] Passing tensors through the Siamese Network...")
    distance_score = siamese_net.predict([current_typing_window, reference_typing_profile])

    print("\n--- Output ---")
    print(f"Euclidean Distance Score: {distance_score[0][0]:.4f}")
    print("\n[Success] The model accepts GAFMAT matrices and outputs a similarity score!")

if __name__ == "__main__":
    main()