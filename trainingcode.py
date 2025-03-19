import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths to dataset folders
train_dir = "archive/train"  # Adjust this path if needed
test_dir = "archive/test"

# Data augmentation and preprocessing
datagen = ImageDataGenerator(rescale=1./255,

                             rotation_range=20,#Rotate randomly by 20 degrees
                             width_shift_range=20,#Randomly Shift Images Horizontily
                             height_shift_range=0.2,#Randomly Shift Images Verticly
                             shear_range=0.2,#shear transformation/(slant Images)
                             zoom_range=0.2,#Zoom in and out ranodmly
                             horizontal_flip=True,#Flip Images left and right
                             fill_mode="nearest"#Fill in the missing pixels after the transformation
                             )  # Normalize pixel values (0-255 -> 0-1)

# Load training images from directory
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),  # Resize all images to 48x48
    batch_size=64,
    color_mode="grayscale",  # Convert images to grayscale
    class_mode="categorical"  # One-hot encode labels
)

# Load testing images from directory
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical"
)

print("Dataset Loaded Successfully!")

from tensorflow.keras.layers import BatchNormalization
# Build the CNN model
model = Sequential([
    # First Conventional block
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),#Normalize activations to Stabalize Training
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),# Drop 30% of nerons to prevent overfitting

    #Second Conventional block
    Conv2D(128, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),  # Normalize activations to stabilize training
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),  # Drop 30% of neurons to prevent overfitting

    # Third Conventional block
    Conv2D(256, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),  # Normalize activations to stabilize training
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),


    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 output classes for emotions
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Display model summary
model.summary()



from tensorflow.keras.callbacks import ReduceLROnPlateau

#Reduce Learning Rate when Accuracy stops improving
lr_scheduler = ReduceLROnPlateau(
    monitor="val_accuracy",# Check Validation Accuracy
    patience=3,#If Accuarcy does not improve for 3 enochs, reduce Learning Rate
    factor=0.5,# Reduce Learning rate by 50%
    min_lr=0.00001 # minimum learning rate
)




# Train the model using image data from directories
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=test_generator,
    callbacks=[lr_scheduler]#Add Learning Rate Schedulizer
)

# Save the trained model
model.save("emotion_model.keras")

print("Model Training Complete. Model Saved as 'emotion_model.h5'.")

# Evaluate model performance
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy*100:.2f}%")

import matplotlib.pyplot as plt

#Plot training and validatiion accuracy
plt.plot(history.history["accuracy"], label = "Training Accuracy")
plt.plot(history.history["val_accuracy"],label = ["validation_accuracy"])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training Vs Validtion Accuracy")
plt.show()

#Plot the Loss
#Plot training and validatiion accuracy
plt.plot(history.history["loss"], label = "Training Loss")
plt.plot(history.history["val_loss"],label = ["Validation Loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Vs Validtion Loss")
plt.show()
