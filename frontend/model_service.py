"""
model_service.py
------------------
Bu dosya, ARAYÜZ ile YAPAY ZEKA MODELİ arasındaki köprüdür.

Şu an model henüz hazır olmadığı için burada SAHTE (mock) sonuçlar üretiyoruz.
Böylece arayüzü baştan sona geliştirip test edebiliriz.

Model / API hazır olduğunda SADECE bu dosyadaki `predict()` fonksiyonunun içini
değiştirmemiz yeterli olacak; app.py'ye (arayüze) hiç dokunmayacağız.

NOT (rapordan): Model her sınıf için AYRI (bağımsız) bir olasılık üretir
(sigmoid). Yani skorların toplamı 1 (yüzde 100) etmez; her kategori kendi
başına değerlendirilir. Bu yüzden arayüzde bunları "yüzde dağılımı" gibi
değil, her kategori için ayrı "olasılık/güven" olarak gösteriyoruz.
"""

from __future__ import annotations
import time
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# 11 tanı kategorisi (rapordaki MILK10k sınıfları). Kullanıcıya görünen
# metinler İngilizce. "group" alanı arayüzde renklendirme için kullanılır.
# ---------------------------------------------------------------------------
CLASSES = [
    {"code": "NV",      "name": "Melanocytic nevus (mole)",                  "group": "Benign"},
    {"code": "BKL",     "name": "Benign keratinocytic lesion",              "group": "Benign"},
    {"code": "DF",      "name": "Dermatofibroma",                           "group": "Benign"},
    {"code": "VASC",    "name": "Vascular lesion / haemorrhage",            "group": "Benign"},
    {"code": "BEN_OTH", "name": "Other benign proliferation",              "group": "Benign"},
    {"code": "INF",     "name": "Inflammatory / infectious condition",      "group": "Inflammatory"},
    {"code": "AKIEC",   "name": "Actinic keratosis / in-situ carcinoma",    "group": "Pre-malignant"},
    {"code": "BCC",     "name": "Basal cell carcinoma",                     "group": "Malignant"},
    {"code": "SCCKA",   "name": "Squamous cell carcinoma / keratoacanthoma","group": "Malignant"},
    {"code": "MEL",     "name": "Melanoma",                                 "group": "Malignant"},
    {"code": "MAL_OTH", "name": "Other malignant proliferation",           "group": "Malignant"},
]

# Vücut bölgesi seçenekleri (İngilizce)
ANATOM_SITES = [
    "Head / neck",
    "Anterior torso",
    "Posterior torso",
    "Upper limb (arm)",
    "Lower limb (leg)",
    "Palms / soles",
    "Oral / genital",
    "Unknown",
]

SEX_OPTIONS = ["Female", "Male", "Unspecified"]


def predict(clinical_img: Image.Image,
            dermoscopic_img: Image.Image,
            metadata: dict) -> dict:
    """
    Bir lezyon için tahmin döndürür.

    GİRDİLER:
      clinical_img    : Klinik (normal) yakın çekim fotoğrafı (PIL Image)
      dermoscopic_img : Dermoskopik fotoğraf (PIL Image)
      metadata        : {"age": int, "sex": str, "skin_tone": int, "site": str}

    ÇIKTI (sözlük):
      {
        "scores": [0.82, 0.13, ...],       # 11 sınıf için BAĞIMSIZ olasılıklar (her biri 0-1)
        "gradcam_clinical":    PIL Image,   # klinik görsel için ısı haritası
        "gradcam_dermoscopic": PIL Image,   # dermoskopik görsel için ısı haritası
        "inference_seconds":   float,       # tahmin süresi
      }

    ============================================================
    ŞU AN: Aşağısı SAHTE veri üretir (demo amaçlı).
    GERÇEK MODELE GEÇERKEN: Bu fonksiyonun içini şu şekilde değiştireceğiz:
        - Görselleri modele/API'ye gönder
        - Dönen 11 skoru ve Grad-CAM ısı haritalarını al
        - Aynı sözlük formatında geri döndür
    Arayüz (app.py) hiç değişmeyecek.
    ============================================================
    """
    start = time.time()

    # --- SAHTE skorlar üret (bağımsız, toplamları 1 OLMAK ZORUNDA DEĞİL) -----
    seed = _seed_from_inputs(clinical_img, metadata)
    rng = np.random.default_rng(seed)
    # Her sınıf için düşük bir taban skor
    scores = rng.random(len(CLASSES)) * 0.35
    # Bir "baskın" sınıf ve bir-iki ikincil sınıfı yükselt (gerçekçi görünsün)
    dominant = rng.integers(0, len(CLASSES))
    scores[dominant] = 0.70 + rng.random() * 0.28
    secondary = rng.integers(0, len(CLASSES))
    scores[secondary] = max(scores[secondary], 0.40 + rng.random() * 0.25)
    scores = [float(min(0.99, s)) for s in scores]

    # --- SAHTE Grad-CAM ısı haritaları üret ---------------------------------
    gradcam_clinical = _fake_heatmap(clinical_img, rng)
    gradcam_dermoscopic = _fake_heatmap(dermoscopic_img, rng)

    elapsed = time.time() - start
    return {
        "scores": scores,
        "gradcam_clinical": gradcam_clinical,
        "gradcam_dermoscopic": gradcam_dermoscopic,
        "inference_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar (sadece sahte demo için; gerçek modelde silinebilir)
# ---------------------------------------------------------------------------
def _seed_from_inputs(img: Image.Image, metadata: dict) -> int:
    """Görsel + yaş bilgisinden tekrarlanabilir bir sayı üretir."""
    small = np.asarray(img.convert("L").resize((16, 16)), dtype=np.int64)
    base = int(small.sum()) + int(metadata.get("age", 0)) * 7
    return base % (2**31)


def _fake_heatmap(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """
    Görselin üzerine, modelin 'baktığı yer' gibi görünen sahte bir sıcak
    bölge (kırmızı-sarı) bindirir. Gerçek Grad-CAM gelince bu fonksiyon
    gerçek ısı haritasıyla değişecek.
    """
    from matplotlib import cm

    base = img.convert("RGB").resize((300, 300))
    arr = np.asarray(base, dtype=np.float32) / 255.0
    h, w = 300, 300

    cy, cx = rng.integers(90, 210), rng.integers(90, 210)
    yy, xx = np.mgrid[0:h, 0:w]
    sigma = rng.integers(45, 80)
    heat = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2)))
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    colored = cm.get_cmap("jet")(heat)[:, :, :3]
    alpha = 0.45 * heat[:, :, None]
    blended = (1 - alpha) * arr + alpha * colored
    blended = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(blended)
