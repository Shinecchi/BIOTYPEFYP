"""
train_model.py
--------------
Trains the Siamese Neural Network (SNN) using Triplet Loss
on the generated (10, 10, 2) GAFMAT images.

USAGE (in Google Colab):
  !python -m src.scripts.train_model --data_dir data/processed/ --batch_size 128 --epochs 50 --output data/biotype_trained_weights.weights.h5
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple

# ---------------------------------------------------------------------------
# 1. Siamese Network Architecture (CNN + LSTM -> 64D Embedding)
# ---------------------------------------------------------------------------
def build_embedding_network(input_shape=(10, 10, 2)) -> Model:
    """
    CNN + LSTM architecture as defined in the FYP proposal Phase 2.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Spatial Features (CNN)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    # Reshape for Temporal Features (LSTM)
    # Shape is now roughly (batch, 2, 2, 128). We flatten spatial dims, keep temporal sequence if needed.
    # In standard GAFMAT, we flatten spatial to sequence:
    shape_dim = x.shape
    new_shape = (shape_dim[1] * shape_dim[2], shape_dim[3])
    x = layers.Reshape(new_shape)(x)
    
    # Temporal Features (LSTM)
    x = layers.LSTM(64, return_sequences=False)(x)
    
    # Final Embedding
    x = layers.Dense(64, activation=None)(x)
    # L2 Normalise embeddings to a hypersphere (vital for Triplet Loss)
    outputs = layers.UnitNormalization(axis=1)(x)
    
    return Model(inputs, outputs, name="embedding_net")


# ---------------------------------------------------------------------------
# 2. Triplet Loss Function
# ---------------------------------------------------------------------------
@tf.function
def semi_hard_triplet_loss(y_true, y_pred, margin=0.5):
    """
    Computes Triplet Loss with Semi-hard negative mining.
    Expects y_pred to contain (anchor, positive, negative) stacked or as batched embeddings.
    Here we use TensorFlow Addons' approach where y_true=labels, y_pred=embeddings.
    However, tfa is deprecated. We implement standard batch hard/semi-hard here using tf operations.
    For simplicity and reliability, we will use a custom training loop with pre-mined triplets.
    """
    pass # We will use a custom layer instead for explicitly passed triplets


class TripletLossLayer(layers.Layer):
    def __init__(self, alpha=0.5, **kwargs):
        super(TripletLossLayer, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs):
        anchor, positive, negative = inputs
        # Euclidian distance
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        
        # max(pos_dist - neg_dist + alpha, 0)
        basic_loss = pos_dist - neg_dist + self.alpha
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
        self.add_loss(loss)
        return loss


def build_siamese_model(base_model: Model, input_shape=(10, 10, 2)) -> Model:
    """
    Builds the 3-input Siamese architecture.
    """
    anchor_input = layers.Input(shape=input_shape, name='anchor')
    positive_input = layers.Input(shape=input_shape, name='positive')
    negative_input = layers.Input(shape=input_shape, name='negative')

    # Pass inputs through the shared embedding network
    emb_a = base_model(anchor_input)
    emb_p = base_model(positive_input)
    emb_n = base_model(negative_input)

    # Triplet Loss Layer
    loss_layer = TripletLossLayer(alpha=0.5)([emb_a, emb_p, emb_n])
    
    siamese_net = Model(inputs=[anchor_input, positive_input, negative_input], outputs=loss_layer)
    siamese_net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return siamese_net


# ---------------------------------------------------------------------------
# 3. Offline Triplet Generation (Data Loader)
# ---------------------------------------------------------------------------
def load_all_users(data_dir: str) -> dict:
    """Loads all .npy files into a dict {user_id: array_of_images}"""
    print(f"Loading data from {data_dir}...")
    user_data = {}
    for fname in os.listdir(data_dir):
        if fname.endswith("_images.npy"):
            user_id = fname.split("_images.npy")[0]
            path = os.path.join(data_dir, fname)
            # Load and ensure shape (N, 10, 10, 2)
            arr = np.load(path)
            user_data[user_id] = arr
    print(f"Loaded {len(user_data)} users.")
    return user_data

def generate_triplets(user_data: dict, triplets_per_user: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates offline (Anchor, Positive, Negative) triplets.
    """
    anchors, positives, negatives = [], [], []
    users = list(user_data.keys())
    
    print("Generating training triplets...")
    for user_id, images in user_data.items():
        n_imgs = len(images)
        if n_imgs < 2:
            continue
            
        for _ in range(triplets_per_user):
            # 1. Pick anchor and positive (same user)
            idx_a, idx_p = np.random.choice(n_imgs, size=2, replace=False)
            
            # 2. Pick negative (different user)
            neg_user_id = np.random.choice([u for u in users if u != user_id])
            neg_images = user_data[neg_user_id]
            idx_n = np.random.randint(0, len(neg_images))
            
            anchors.append(images[idx_a])
            positives.append(images[idx_p])
            negatives.append(neg_images[idx_n])

    return np.array(anchors), np.array(positives), np.array(negatives)


# ---------------------------------------------------------------------------
# 4. Main Training Orchestrator
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Train BioType Siamese Network')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with .npy files')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--output', type=str, default='data/biotype_trained_weights.weights.h5')
    args = parser.parse_args()

    # Avoid OOM by allowing memory growth on GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("🚀 GPU Detected! Hardware acceleration enabled.")
        except RuntimeError as e:
            print(e)
    else:
        print("⚠️ No GPU detected. Training will run on CPU and be very slow.")

    # 1. Load Data
    user_data = load_all_users(args.data_dir)
    
    # Generate 1500 triplets per user (For 99 users = ~148,500 triplets)
    a, p, n = generate_triplets(user_data, triplets_per_user=1500)
    print(f"Generated {len(a)} total triplets for training.")

    # Split into train/val (90/10)
    split = int(0.9 * len(a))
    a_train, p_train, n_train = a[:split], p[:split], n[:split]
    a_val, p_val, n_val = a[split:], p[split:], n[split:]

    # 2. Build Model
    base_encoder = build_embedding_network(input_shape=(10, 10, 2))
    siamese_model = build_siamese_model(base_encoder, input_shape=(10, 10, 2))
    
    print("\nModel Architecture built successfully.")
    
    # 3. Train
    print("\nStarting Training Setup...")
    # Dummy targets needed for Keras API since loss is inside layer
    dummy_y_train = np.zeros((len(a_train), 1))
    dummy_y_val = np.zeros((len(a_val), 1))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]

    print("\n==================================")
    print("      IGNITION: COMMENCE TRAINING ")
    print("==================================\n")

    siamese_model.fit(
        x=[a_train, p_train, n_train],
        y=dummy_y_train,
        validation_data=([a_val, p_val, n_val], dummy_y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # 4. Save EXACTLY what is needed for Inference 
    # We only save the weights of the `base_encoder`! The 3-branch siamese wrapper is only for training.
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    base_encoder.save_weights(args.output)
    print(f"\n✅ Training Complete. Base Embedder weights saved to: {args.output}")

if __name__ == '__main__':
    main()
