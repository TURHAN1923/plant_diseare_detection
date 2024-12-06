import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Test görüntüsünün yolu
image_path = r'C:\proje_test\corn-leaf-8253_1280.jpg'

# Modeli yükle
model = tf.keras.models.load_model("plant_disease_model.h5")

# Görüntüyü yükle ve yeniden boyutlandır
test_image = cv2.imread(image_path)
test_image_resized = cv2.resize(test_image, (224, 224))

# Görüntüyü normalize et ve şekillendir
test_image_normalized = test_image_resized / 255.0
test_image_expanded = np.expand_dims(test_image_normalized, axis=0)  # Model için batch boyutu ekle

# Model ile tahmin yap
predicted_label = model.predict(test_image_expanded)
predicted_class = np.argmax(predicted_label, axis=1)

# Etiket isimleri (gerekirse eğitim setine göre düzenleyin)
class_names = ['Healthy', 'Diseased']  # Sınıf isimlerini kendi verinize göre ayarlayın

# Tahmini yazdır
print(f"Modelin Tahmini: {class_names[predicted_class[0]]}")

# Görüntüyü göster ve tahmini başlık olarak ekle
plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
plt.title(f"Tahmin: {class_names[predicted_class[0]]}")
plt.axis('off')
plt.show()
