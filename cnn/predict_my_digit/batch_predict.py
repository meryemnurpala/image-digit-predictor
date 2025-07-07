import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Model ve resimlerin yolu
MODEL_PATH = os.path.join("..", "model.h5")
IMAGE_DIR = os.path.join("..")
OUTPUT_DIR = os.path.dirname(__file__)

# Tahmin edilecek resim dosyaları
image_files = [f"resim{i}.png" for i in range(1, 6)]

y_true = []
y_pred = []

# Modeli yükle
model = tf.keras.models.load_model(MODEL_PATH)

for img_file in image_files:
    img_path = os.path.join(IMAGE_DIR, img_file)
    # Görseli oku ve işle
    img = Image.open(img_path).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Tahmin yap
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    y_pred.append(predicted_digit)

    # Gerçek değeri dosya adından al
    true_digit = int(img_file.replace("resim", "").replace(".png", ""))
    y_true.append(true_digit)

    # Tahmin edilen rakamı matplotlib ile görselleştir
    plt.figure(figsize=(2,2))
    plt.imshow(np.ones((28,28)), cmap="gray")  # Siyah arka plan
    plt.text(14, 18, str(predicted_digit), fontsize=32, ha='center', va='center', color='white', weight='bold')
    plt.axis('off')
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"tahmin_{img_file}")
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Accuracy hesapla
accuracy = np.mean(np.array(y_true) == np.array(y_pred))
print(f"Toplam {len(y_true)} görselde doğruluk (accuracy): {accuracy*100:.2f}%")
print("Gerçekler:", y_true)
print("Tahminler:", y_pred)

# Her tahminin doğru/yanlış olduğunu yazdır
for idx, (t, p) in enumerate(zip(y_true, y_pred), 1):
    durum = "Doğru" if t == p else "Yanlış"
    print(f"resim{idx}.png: Gerçek={t}, Tahmin={p} -> {durum}")

# Doğru/yanlış grafiği
results = [t == p for t, p in zip(y_true, y_pred)]
plt.figure(figsize=(8,2))
plt.bar(range(1, len(results)+1), results, color=["green" if r else "red" for r in results])
plt.xlabel("Görsel No")
plt.ylabel("Doğruluk")
plt.title("Her Görsel İçin Tahmin Sonucu (1=Doğru, 0=Yanlış)")
plt.ylim(0, 1.1)
plt.yticks([0,1], ["Yanlış", "Doğru"])
plt.xticks(range(1, len(results)+1))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tahmin_dogruluk_grafigi.png"))
plt.close() 