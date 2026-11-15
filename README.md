# CIFAR-10 Görüntü Sınıflandırma — ResNet18 ile Transfer Öğrenme

**BIL485 Derin Öğrenme — Ödev 2: Computer Vision**

Bu projede açık kaynak **CIFAR-10** veri seti (10 sınıf: uçak, otomobil, kuş, kedi, geyik, köpek, kurbağa, at, gemi, kamyon) üzerinde, **ImageNet'te ön-eğitilmiş ResNet18** modeli ince ayar (fine-tuning) ile eğitilmiştir. Model; Accuracy, Precision, Recall, F1, karışıklık matrisi, öğrenme eğrileri, eğitim ve çıkarım süresi ile değerlendirilmiş, telefonla çekilmiş 5 gerçek fotoğrafla test edilmiş ve **Gradio** ile web arayüzü olarak sunulmuştur.

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `BIL485_Odev2_CIFAR10_ResNet18.ipynb` | Tüm akış: veri, model, eğitim, değerlendirme, telefon testi, Gradio, rapor özeti |
| `app.py` | `best_model.pth` dosyasını yükleyen bağımsız Gradio uygulaması |
| `requirements.txt` | Gerekli Python paketleri |
| `Rapor_Sablonu_Taslak.md` | Ödev raporu taslağı |

## Notebook'u çalıştırma (Google Colab)

1. Notebook'u Colab'da aç.
2. **Runtime > Change runtime type > T4 GPU** seç.
3. **Runtime > Run all**. (Toplam ~15–20 dk.)
4. Yükleme butonu çıkınca telefonla çektiğin 5 fotoğrafı seç (`foto1.jpg … foto5.jpg` diye adlandırmak önerilir).
5. `GERCEK_ETIKETLER` sözlüğüne gerçek sınıfları yaz, o hücreyi tekrar çalıştır.
6. Gradio hücresindeki `…gradio.live` linkinden arayüzü dene.
7. Son hücre `sonuclar.zip` dosyasını indirir (grafikler, metrikler, `rapor_ozeti.txt`, `best_model.pth`).

## Gradio uygulamasını yerelde çalıştırma

```bash
pip install -r requirements.txt
# sonuclar.zip içindeki best_model.pth dosyasını app.py ile aynı klasöre koy
python app.py            # http://127.0.0.1:7860
python app.py --share    # geçici public link
```

> Not: `best_model.pth` ~45 MB'tır; GitHub'ın 100 MB sınırının altında olduğu için repoya doğrudan eklenebilir.

## Yöntem (özet)

- **Veri bölünmesi:** 50.000 eğitim görüntüsü sabit tohumla (seed=42) 45.000 train / 5.000 validation; resmi 10.000 test seti.
- **Ön işleme:** 32×32 → 128×128; train'de RandomResizedCrop(ölçek 0.7–1.0) + yatay çevirme; ImageNet normalizasyonu.
- **Model:** ResNet18 (ImageNet1K_V1), son katman 512→10, tüm katmanlar eğitilir.
- **Eğitim:** AdamW (lr=1e-3, weight decay=0.05), cosine LR planı, batch 128, 4 epoch, CrossEntropyLoss, karma hassasiyet (AMP).
- **Çıkış:** logitler üzerinde softmax → sınıf olasılıkları; arayüzde en olası 3 sınıf.

## Sonuçlar

Notebook sonundaki **RAPOR ÖZETİ** bloğundan doldurulacak:

| Metrik | Değer |
|---|---|
| Test Accuracy | _…_ |
| Precision (macro) | _…_ |
| Recall (macro) | _…_ |
| F1 (macro) | _…_ |
| En iyi Val Accuracy (epoch) | _…_ |
| Eğitim süresi | _…_ |
| Çıkarım süresi (batch=1) | _… ms/görüntü_ |
| Verim (batch=256) | _… görüntü/s_ |
| Telefon fotoğrafları (top-1) | _…/5_ |

![Öğrenme eğrileri](sonuclar/learning_curves.png)
![Karışıklık matrisi](sonuclar/confusion_matrix.png)
![Telefon testi](sonuclar/phone_test_grid.png)

## Kaynak

- CIFAR-10: A. Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009 — https://www.cs.toronto.edu/~kriz/cifar.html
- K. He vd., *Deep Residual Learning for Image Recognition*, CVPR 2016.
