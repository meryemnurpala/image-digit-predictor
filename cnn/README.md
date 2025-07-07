# CNN El Yazısı Rakam Tanıma Projesi

## batch_predict.py Ne Yapar?

Bu script, `cnn` klasöründeki `resim1.png` - `resim5.png` dosyalarını eğitilmiş model ile tahmin eder. Her bir görsel için:
- Model tahminini alır.
- Tahmin edilen rakamı matplotlib ile görselleştirip `predict_my_digit` klasörüne kaydeder.
- Her tahminin doğru/yanlış olduğunu ekrana yazar.
- Tüm tahminler için doğruluk (accuracy) oranını hesaplar ve ekrana yazar.
- Sonuçların özetini gösteren bir çubuk grafiği oluşturur.

---

## What does batch_predict.py do?

This script predicts the digits in `resim1.png` to `resim5.png` using the trained model. For each image:
- Gets the model's prediction.
- Visualizes the predicted digit with matplotlib and saves it to the `predict_my_digit` folder.
- Prints whether each prediction is correct or not.
- Calculates and prints the overall accuracy.
- Creates a bar chart summarizing the results. 