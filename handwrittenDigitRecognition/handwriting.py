import sys
import tensorflow as tf

# Use MNIST handwriting dataset
# MNIST is a dataset of 70,000 handwritten digits (0–9) divided into:
# Training set: 60,000 samples.
# Testing set: 10,000 samples.
mnist = tf.keras.datasets.mnist

# Prepare data for training
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalizes pixel values to the range [0, 1] by dividing by 255.
# Improves the performance of the neural network by making the data smaller and consistent.
x_train, x_test = x_train / 255.0, x_test / 255.0
# Converts numeric labels (e.g., 3) into a binary array:
# Example: 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
# Necessary for classification tasks since the output layer of the model produces probabilities for all classes (10 digits).
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
# Reshape to 4D:
# Neural networks for image data (like CNNs) require input in 4D:
# Shape: (samples, height, width, channels).
# Since the images are grayscale, the channel dimension is 1.
x_train = x_train.reshape(
    x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
)
x_test = x_test.reshape(
    x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
)


# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,  # Rotate images slightly
    zoom_range=0.1,     # Slight zoom
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(x_train)


# Create a convolutional neural network
model = tf.keras.models.Sequential([

    # Convolutional layer. Learn 32 filters using a 3x3 kernel
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    # Batch normalization can stabilize training and improve feature learning
    tf.keras.layers.BatchNormalization(),


    # Max-pooling layer, using 2x2 pool size (from cnn)
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten units (from cnn)
    tf.keras.layers.Flatten(),

    # Add a hidden layer with dropout
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    # Add an output layer with output units for all 10 digits
    # softmax is going tto take output and turn it to probability distribution
    tf.keras.layers.Dense(10, activation="softmax")
])

# Train neural network
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# not using data augmentation
#model.fit(x_train, y_train, epochs=12)
model.fit(datagen.flow(x_train, y_train, batch_size=32), validation_data=(x_test, y_test), epochs=12)


# Evaluate neural network performance
model.evaluate(x_test,  y_test, verbose=2)

# Save model to file
# sys.argv[0] contains the script name.
# sys.argv[1] contains a filename (the user's input).
if len(sys.argv) == 2:
    filename = sys.argv[1] # represents the filename provided by the user.
    if not filename.endswith(".keras"):
        filename += ".keras"
    model.save(filename)
    print(f"Model saved to {filename}.")

# run this tto save model
#python handwriting.py model2_filename.keras