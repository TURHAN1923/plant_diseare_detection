import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, ReLU, InputLayer
import os

# Veri kümesinin yolu
dataset_path = r'D:\\plant'

# Veri artırma
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,  # %20 validation verisi
    rotation_range=10,
    width_shift_range=0.03,
    height_shift_range=0.03
)

# Eğitim ve doğrulama veri yükleyicileri
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Modelin tanımlanması
model = Sequential([
    InputLayer(input_shape=(224, 224, 3)),
    Conv2D(16, (3, 3), padding='same'),
    BatchNormalization(),
    ReLU(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(2, activation='softmax')  # 2 sınıf: Hasta ve sağlıklı
])

# Modelin derlenmesi
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Modelin eğitilmesi
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    shuffle=True
)

# Modelin kaydedilmesi
model.save("plant_disease_model.h5")
