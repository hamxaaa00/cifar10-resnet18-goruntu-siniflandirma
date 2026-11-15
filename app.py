"""
BIL485 Derin Öğrenme — Ödev 2
CIFAR-10 sınıflandırıcı (ResNet18, transfer öğrenme) için bağımsız Gradio uygulaması.

Kullanım:
    pip install -r requirements.txt
    python app.py                 # http://127.0.0.1:7860
    python app.py --share         # geçici public link (…gradio.live)
    python app.py --model yol/best_model.pth

best_model.pth dosyası notebook tarafından üretilir (sonuclar.zip içinde de bulunur).
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import models, transforms
import gradio as gr

CLASS_TR = {"airplane": "uçak", "automobile": "otomobil", "bird": "kuş", "cat": "kedi",
            "deer": "geyik", "dog": "köpek", "frog": "kurbağa", "horse": "at",
            "ship": "gemi", "truck": "kamyon"}


def load_model(path, device):
    """Notebook'ta kaydedilen checkpoint'i (state_dict + meta bilgiler) yükler."""
    ckpt = torch.load(path, map_location=device)
    class_names = ckpt.get("class_names", list(CLASS_TR))
    model = models.resnet18(weights=None)          # ağırlıklar checkpoint'ten gelecek
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    img_size = ckpt.get("img_size", 128)
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(ckpt.get("mean", [0.485, 0.456, 0.406]),
                             ckpt.get("std", [0.229, 0.224, 0.225])),
    ])
    return model, tf, class_names


def build_demo(model, tf, class_names, device):
    @torch.no_grad()
    def predict(img):
        if img is None:
            return {}
        img = ImageOps.exif_transpose(img).convert("RGB")   # telefon fotoğrafı döndürmesini düzelt
        x = tf(img).unsqueeze(0).to(device)
        probs = F.softmax(model(x), dim=1)[0].cpu()          # logit -> olasılık
        p, idx = probs.topk(3)
        return {f"{class_names[i]} ({CLASS_TR.get(class_names[i], class_names[i])})": float(v)
                for v, i in zip(p.tolist(), idx.tolist())}

    return gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Görüntü yükle"),
        outputs=gr.Label(num_top_classes=3, label="En olası 3 sınıf"),
        title="CIFAR-10 Sınıflandırıcı — ResNet18 (Transfer Öğrenme)",
        description=("Bir görüntü yükleyin; model 10 CIFAR-10 sınıfından en olası 3 tanesini "
                     "olasılıklarıyla gösterir. Sınıflar: uçak, otomobil, kuş, kedi, geyik, köpek, "
                     "kurbağa, at, gemi, kamyon."),
    )


def main():
    ap = argparse.ArgumentParser(description="CIFAR-10 ResNet18 Gradio uygulaması")
    ap.add_argument("--model", default="best_model.pth", help="checkpoint yolu")
    ap.add_argument("--share", action="store_true", help="public Gradio linki oluştur")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Model dosyası bulunamadı: {args.model} (notebook'u çalıştırıp best_model.pth'yi buraya kopyalayın)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tf, class_names = load_model(args.model, device)
    print(f"Model yüklendi ({device}).")
    build_demo(model, tf, class_names, device).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
