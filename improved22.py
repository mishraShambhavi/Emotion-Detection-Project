import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Parameters
IMG_SIZE = 48
NUM_CLASSES = 7  # Adjust to 7 if adding another emotion
EPOCHS = 25
BATCH_SIZE = 16
TARGET_EMOTIONS = ['happy', 'sad', 'neutral', 'suprise', 'contempt', 'disgust','angry']  # Add 'anger' for 7 classes if needed

# 1. Data Preprocessing with LBP and ORB
def load_and_preprocess_data(data_dir):
    images = []
    labels = []
    grid_size = 8

    for emotion in TARGET_EMOTIONS:
        folder_path = os.path.join(data_dir, emotion)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found, skipping.")
            continue
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img_normalized = img.astype('float32') / 255.0

                lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
                lbp_normalized = lbp / (np.max(lbp) + 1e-6)

                orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
                orb_map = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
                step = IMG_SIZE // grid_size
                for i in range(0, IMG_SIZE, step):
                    for j in range(0, IMG_SIZE, step):
                        region = img[i:i+step, j:j+step]
                        keypoints, descriptors = orb.detectAndCompute(region, None)
                        if keypoints and descriptors is not None:
                            strongest_kp = max(keypoints, key=lambda k: k.response)
                            x, y = int(strongest_kp.pt[0] + j), int(strongest_kp.pt[1] + i)
                            if 0 <= x < IMG_SIZE and 0 <= y < IMG_SIZE:
                                orb_map[y, x] = np.mean(descriptors[0])
                orb_normalized = orb_map / (np.max(orb_map) + 1e-6)

                def z_score_normalize(feature, K=100, C=1e-6):
                    mean = np.mean(feature)
                    std = np.std(feature) + C
                    return K * (feature - mean) / std

                img_z = z_score_normalize(img_normalized)
                lbp_z = z_score_normalize(lbp_normalized)
                orb_z = z_score_normalize(orb_normalized)

                img_combined = np.stack([img_z, lbp_z, orb_z], axis=-1)
                images.append(img_combined)
                labels.append(TARGET_EMOTIONS.index(emotion))
    return np.array(images), tf.keras.utils.to_categorical(labels, NUM_CLASSES)

# Load data
train_dir = r"C:\Users\shambhavi.mishra-st\Desktop\ferplus\train"
val_dir = r"C:\Users\shambhavi.mishra-st\Desktop\ferplus\validation"
test_dir = r"C:\Users\shambhavi.mishra-st\Desktop\ferplus\test"

X_train, y_train = load_and_preprocess_data(train_dir)
X_val, y_val = load_and_preprocess_data(val_dir)
X_test, y_test = load_and_preprocess_data(test_dir)

print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

# 2. New CNN Architecture
def build_convnet():
    model = Sequential([
        # 1 - Convolution
        layers.Conv2D(128, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),  # Adjusted for 3 channels
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Dropout(0.25),

        # 2nd Convolution layer
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Dropout(0.25),

        # 3rd Convolution layer
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Dropout(0.25),

        # 4th Convolution layer
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Dropout(0.25),

        # Flattening
        layers.Flatten(),

        # Fully connected layer 1st layer
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.25),

        # Fully connected layer 2nd layer
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.25),

        # Output layer
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Build and compile the model
model = build_convnet()
optimizer = Adam(learning_rate=0.0005)  # From new architecture
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 3. Train the Model
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    verbose=1,
    shuffle=True
)

# 4. Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=TARGET_EMOTIONS)
disp.plot(cmap='YlGnBu')
plt.title("Confusion Matrix")
plt.show()

# 5. Plot Training History
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# 6. Save the Model
model.save('FourConvNet.h5')
print("Model saved as 'FourConvNet.h5'")