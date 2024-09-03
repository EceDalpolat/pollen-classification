import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# Dosya yolları
images_dir = r"C:\Users\Lenovo\OneDrive\Desktop\dataset\images"  # Görsellerin bulunduğu dizin
labels_csv = r"C:\Users\Lenovo\OneDrive\Desktop\dataset\bboxes2.csv"  # Etiketlerin bulunduğu CSV dosyası

# CSV dosyasını sütun isimleri ile yükleme
labels_df = pd.read_csv(labels_csv, names=['filename', 'label'], header=None)

# Veriyi %80 eğitim ve %20 test olarak ayırma
train_df, test_df = train_test_split(labels_df, test_size=0.2, random_state=42)

# Görsel veri yükleyici ayarları
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Eğitim veri yükleyici
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=images_dir,
    x_col='filename',  # CSV dosyasındaki görsel isimleri
    y_col='label',  # CSV dosyasındaki etiketler
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Test veri yükleyici
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=images_dir,
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Modeli oluşturma
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğitme
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)

# Modeli değerlendirme
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Modeli kaydetme
model.save('polen_siniflandirma_cnn.h5')

# Sınıf isimlerini almak
class_names = list(train_generator.class_indices.keys())

# Yeni veri setini toplu olarak sınıflandırma
new_images_dir = r"C:\Users\Lenovo\OneDrive\Desktop\images"

# Desteklenen dosya uzantıları
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

# Tüm görüntüleri yükleyip, tahmin etmek için
for img_name in os.listdir(new_images_dir):
    if os.path.splitext(img_name)[1].lower() in valid_extensions:  # Dosya uzantısını kontrol et
        img_path = os.path.join(new_images_dir, img_name)

        try:
            # Görüntüyü yükleme ve modelin beklediği formata dönüştürme
            img = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # (1, 150, 150, 3) şekline getirir
            img_array /= 255.  # Normalizasyon, modelin eğitildiği gibi

            # Tahmin yapma
            predictions = model.predict(img_array)

            # En yüksek olasılığa sahip sınıfı bulma
            predicted_class = class_names[np.argmax(predictions)]
            confidence_score = np.max(predictions)

            print(f"Image: {img_name}, Predicted Class: {predicted_class}, Confidence: {confidence_score:.4f}")

        except Exception as e:
            print(f"Error processing image {img_name}: {e}")
