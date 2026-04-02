"""
authenticator.py
-----------------
The AI Brain of BioType.

Loads the trained Siamese Network (CNN+LSTM) from weights on disk,
performs user enrollment (building a prototype embedding), and
live verification (comparing current typing against the prototype).

This module consumes (10, 10, 2) GAFMAT image tensors and outputs
Euclidean distances in the 64-dimensional embedding space.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# Default weights path (relative to project root)
DEFAULT_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), 'model', 'biotype_trained_weights.weights.h5'
)


# ---------------------------------------------------------------------------
# Network Architecture (MUST match train_model.py exactly)
# ---------------------------------------------------------------------------

def _build_embedding_network(input_shape=(10, 10, 2)) -> Model:
    """
    CNN + LSTM architecture — identical to the one used during training.
    Only the BASE ENCODER is needed for inference.
    """
    inputs = layers.Input(shape=input_shape)

    # Spatial Features (CNN)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Reshape for LSTM
    shape_dim = x.shape
    new_shape = (shape_dim[1] * shape_dim[2], shape_dim[3])
    x = layers.Reshape(new_shape)(x)

    # Temporal Features (LSTM)
    x = layers.LSTM(64, return_sequences=False)(x)

    # Embedding Head
    x = layers.Dense(64, activation=None)(x)
    outputs = layers.UnitNormalization(axis=1)(x)

    return Model(inputs, outputs, name="embedding_net")


# ---------------------------------------------------------------------------
# BioTypeAuthenticator
# ---------------------------------------------------------------------------

class BioTypeAuthenticator:
    """
    Handles enrollment and continuous verification.

    Enrollment: Takes several GAFMAT images of the registered user,
                runs them through the encoder, and averages them into
                a single 64D prototype embedding.

    Verification: Takes a single new GAFMAT image, encodes it, and
                  computes the Euclidean distance to the prototype.
    """

    def __init__(self, weights_path: str = DEFAULT_WEIGHTS_PATH):
        weights_path = os.path.abspath(weights_path)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Model weights not found at: {weights_path}\n"
                f"Please ensure 'biotype_trained_weights.weights.h5' is in src/model/"
            )

        self._encoder = _build_embedding_network(input_shape=(10, 10, 2))
        self._encoder.load_weights(weights_path)
        # Suppress TF logs during inference
        tf.get_logger().setLevel('ERROR')

        self._prototype: np.ndarray | None = None  # (64,) enrolled user embedding
        print(f"[BioTypeAuthenticator] Model loaded from {weights_path}")

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def enroll(self, enrollment_images: list[np.ndarray]) -> bool:
        """
        Build a user prototype from a list of GAFMAT enrollment images.

        Args:
            enrollment_images : List of np.ndarray, each shape (10, 10, 2).
                               Recommended: 5–20 images for a stable prototype.

        Returns:
            True if enrollment succeeded.
        """
        if len(enrollment_images) == 0:
            print("[Authenticator] Enrollment failed: no images provided.")
            return False

        batch = np.stack(enrollment_images, axis=0)   # (N, 10, 10, 2)
        embeddings = self._encoder.predict(batch, verbose=0)  # (N, 64)

        # Prototype = centroid of all enrollment embeddings
        self._prototype = np.mean(embeddings, axis=0)   # (64,)
        # Re-normalise the mean vector
        norm = np.linalg.norm(self._prototype)
        if norm > 0:
            self._prototype /= norm

        print(f"[Authenticator] Enrolled successfully ({len(enrollment_images)} images used).")
        return True

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self, gafmat_image: np.ndarray) -> float:
        """
        Compute the Euclidean distance between a new image and the enrolled prototype.

        Args:
            gafmat_image : np.ndarray of shape (10, 10, 2).

        Returns:
            Euclidean distance (float). Lower = more similar to enrolled user.
            Raises RuntimeError if not enrolled yet.
        """
        if self._prototype is None:
            raise RuntimeError("User not enrolled. Call enroll() first.")

        img = gafmat_image[np.newaxis, ...]             # (1, 10, 10, 2)
        embedding = self._encoder.predict(img, verbose=0)[0]  # (64,)

        distance = float(np.linalg.norm(embedding - self._prototype))
        return distance

    @property
    def is_enrolled(self) -> bool:
        return self._prototype is not None
