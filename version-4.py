import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dosya yolları
images_dir = r"C:\Users\Lenovo\OneDrive\Desktop\dataset\images"
labels_csv = r"C:\Users\Lenovo\OneDrive\Desktop\dataset\bboxes2.csv"

# CSV dosyasını sütun isimleri ile yükleme
labels_df = pd.read_csv(labels_csv, names=['filename', 'label'], header=None)

# filename sütununu string formatına dönüştürme
labels_df['filename'] = labels_df['filename'].astype(str)

# Eğer label sütununda da dönüşüm gerekiyorsa (örneğin, sayılar string olmalıysa)
labels_df['label'] = labels_df['label'].astype(str)
# Görsel veri yükleyici ayarları
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Veri yükleyiciler
train_generator = train_datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory=images_dir,
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory=images_dir,
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# DenseNet121 modeli
densenet_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
densenet_model.trainable = False  # Transfer learning için önceden eğitilmiş ağırlıkları dondur

model_2 = models.Sequential([
    densenet_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

model_2.compile(optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
import tensorflow as tf
# print("Train samples:", train_generator.samples)
# print("Test samples:", test_generator.samples)
# print(labels_df.head())
# print(train_generator.class_indices)

# Modeli yüklemek
model = tf.keras.models.load_model('polen_siniflandirma_cnn.h5')
# Modeli yeniden derle
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# CNN modeli eğitme
history_cnn =model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)


# DenseNet121 modeli eğitme
history_densenet = model_2.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)
# CNN model değerlendirmesi
loss_cnn, accuracy_cnn = model.evaluate(test_generator)
print(f"CNN Model - Test Loss: {loss_cnn:.4f}, Test Accuracy: {accuracy_cnn:.4f}")

# DenseNet121 model değerlendirmesi
loss_densenet, accuracy_densenet = model_2.evaluate(test_generator)
print(f"DenseNet121 Model - Test Loss: {loss_densenet:.4f}, Test Accuracy: {accuracy_densenet:.4f}")

# SGD optimizer ile DenseNet121 modeli
model_2.compile(optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.9),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

history_densenet_sgd = model_2.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)


# SGD optimizer ile CNN modeli
model.compile(optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.9),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

history_CNN_sgd = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)

import matplotlib.pyplot as plt

# Model evaluation
loss_cnn_adam, accuracy_cnn_adam = model.evaluate(test_generator)
loss_densenet_adam, accuracy_densenet_adam = model_2.evaluate(test_generator)

# Plotting training history
plt.figure(figsize=(12, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history_cnn.history['accuracy'], label='CNN - Adam')
plt.plot(history_densenet.history['accuracy'], label='DenseNet121 - Adam')
plt.plot(history_CNN_sgd.history['accuracy'], label='CNN - SGD')
plt.plot(history_densenet_sgd.history['accuracy'], label='DenseNet121 - SGD')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history_cnn.history['loss'], label='CNN - Adam')
plt.plot(history_densenet.history['loss'], label='DenseNet121 - Adam')
plt.plot(history_CNN_sgd.history['loss'], label='CNN - SGD')
plt.plot(history_densenet_sgd.history['loss'], label='DenseNet121 - SGD')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Print final evaluation results
print(f"CNN Model (Adam) - Test Loss: {loss_cnn_adam:.4f}, Test Accuracy: {accuracy_cnn_adam:.4f}")
print(f"DenseNet121 Model (Adam) - Test Loss: {loss_densenet_adam:.4f}, Test Accuracy: {accuracy_densenet_adam:.4f}")

