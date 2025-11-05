# Task 2: Deep Learning with TensorFlow
# Dataset: MNIST Handwritten Digits
# Goal: Build CNN >95% accuracy + visualize predictions

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and preprocess MNIST data
# Load the MNIST dataset (60k train + 10k test images, 28x28 grayscale)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape to add channel dimension: (28, 28) → (28, 28, 1)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Convert labels to one-hot encoding (optional, but good practice)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Step 2: Build CNN model
model = models.Sequential([
    # First Conv Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Second Conv Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third Conv Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flatten and Dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Prevent overfitting
    layers.Dense(10, activation='softmax')  # 10 classes (digits 0-9)
])

# Display model architecture
model.summary()

# Step 3: Compile and train model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=5,  # Usually enough to exceed 95% on MNIST
    validation_data=(x_test, y_test),
    verbose=1
)

# Step 4: Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# Ensure we meet the goal
assert test_acc > 0.95, "Model did not achieve >95% accuracy!"

# Step 5: Visualize predictions on 5 sample images
# Get raw predictions (probabilities)
predictions = model.predict(x_test[:5])

# Convert one-hot back to integers for true labels
y_test_labels = np.argmax(y_test, axis=1)

# Plot images with true and predicted labels
plt.figure(figsize=(12, 4))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    pred_label = np.argmax(predictions[i])
    true_label = y_test_labels[i]
    color = 'green' if pred_label == true_label else 'red'
    plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
    plt.axis('off')
plt.tight_layout()
plt.show()