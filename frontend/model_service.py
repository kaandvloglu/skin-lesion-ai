"""
model_service.py
------------------
Bu dosya, ARAYÜZ ile YAPAY ZEKA MODELİ arasındaki köprüdür.

Model artık backend (Spring Boot → FastAPI → TFLite) üzerinden internette
yayında, o yüzden burada GERÇEK tahmin isteği atıyoruz:
    Frontend → Spring Boot /api/predictions/upload → (FastAPI + model) → cevap

Giriş çerezi gerekiyorsa diye, auth_service ile AYNI oturumu (requests.Session)
kullanıyoruz.

GÜVENLİK ANAHTARI:
    USE_BACKEND = True iken gerçek modele bağlanır.
    Sorun olursa tek satırı `USE_BACKEND = False` yap → uygulama anında eski
    çalışan SAHTE (mock) sonuçlarına döner. Canlı uygulama asla bozulmaz.

NOT (rapordan): Model her sınıf için AYRI (bağımsız) bir olasılık üretir
(sigmoid). Skorların toplamı %100 etmez; her kategori kendi başına değerlendirilir.
"""

from __future__ import annotations
import io
import time
import base64

import numpy as np
from PIL import Image

import auth_service

# ---------------------------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------------------------
USE_BACKEND = True
BACKEND_URL = "https://skin-lesion-backend-nnfv.onrender.com"
PREDICT_PATH = "/api/predictions/upload"
GRADCAM_PATH = "/api/predictions/gradcam"  # Grad-CAM ayrı endpoint (base64 döner)
TIMEOUT = 120  # Render ilk istekte "uykudan" uyanabilir → geniş zaman aşımı

# ---------------------------------------------------------------------------
# 11 tanı kategorisi (MILK10k). Kullanıcıya görünen metinler İngilizce.
# "code" değerleri backend'in döndürdüğü skor anahtarlarıyla BİREBİR aynı.
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

# ---------------------------------------------------------------------------
# Vücut bölgesi seçenekleri.
# Arayüzde okunaklı İngilizce etiket gösterilir; backend'e ise onun beklediği
# kısa kod ("trunk", "head_neck_face" ...) gönderilir. SITE_API bu eşlemeyi tutar.
# Backend'in kabul ettiği 7 değer: foot, genital, hand, head_neck_face,
# lower_extremity, trunk, upper_extremity.
# ---------------------------------------------------------------------------
SITE_API = {
    "Head / neck / face":  "head_neck_face",
    "Trunk":               "trunk",
    "Upper limb (arm)":    "upper_extremity",
    "Lower limb (leg)":    "lower_extremity",
    "Hand":                "hand",
    "Foot":                "foot",
    "Genital":             "genital",
}
ANATOM_SITES = list(SITE_API.keys())

# Cinsiyet: arayüzde okunaklı etiket, backend'e küçük harf değer.
# Backend'in kabul ettiği değerler: male, female.
SEX_API = {
    "Female": "female",
    "Male":   "male",
}
SEX_OPTIONS = list(SEX_API.keys())


def predict(clinical_img: Image.Image,
            dermoscopic_img: Image.Image,
            metadata: dict) -> dict:
    """
    Bir lezyon için tahmin döndürür.

    GİRDİLER:
      clinical_img    : Klinik yakın çekim (PIL Image)
      dermoscopic_img : Dermoskopik fotoğraf (PIL Image)
      metadata        : {"age": int, "sex": str, "skin_tone": int, "site": str}
                        (sex ve site burada ARAYÜZ ETİKETİ olarak gelir;
                         backend'e gönderirken kısa koda çevrilir.)

    ÇIKTI (sözlük):
      {
        "scores": [11 bağımsız olasılık, CLASSES sırasında],
        "gradcam_clinical":    PIL Image | None,
        "gradcam_dermoscopic": PIL Image | None,
        "inference_seconds":   float,
      }
    """
    start = time.time()

    # ----- GERÇEK BACKEND -----
    if USE_BACKEND:
        result = _predict_backend(clinical_img, dermoscopic_img, metadata)
        result["inference_seconds"] = time.time() - start
        return result

    # ----- MOCK (yedek) -----
    result = _predict_mock(clinical_img, dermoscopic_img, metadata)
    result["inference_seconds"] = time.time() - start
    return result


# ---------------------------------------------------------------------------
# GERÇEK backend tahmini
# ---------------------------------------------------------------------------
def _predict_backend(clinical_img, dermoscopic_img, metadata) -> dict:
    session = auth_service.get_session()

    # 1) TAHMİN — /api/predictions/upload
    files, data = _build_multipart(clinical_img, dermoscopic_img, metadata)
    r = session.post(f"{BACKEND_URL}{PREDICT_PATH}", files=files, data=data, timeout=TIMEOUT)
    r.raise_for_status()
    payload = r.json()

    # Skorları CLASSES sırasına göre listeye çevir
    raw_scores = payload.get("scores", {}) or {}
    scores = [float(raw_scores.get(cls["code"], 0.0)) for cls in CLASSES]

    # 2) GRAD-CAM — ayrı endpoint /api/predictions/gradcam (base64 döner).
    #    En iyi çaba: başarısız olursa None döner, tahmin yine gösterilir.
    gradcam_clinical, gradcam_dermoscopic = _fetch_gradcam(
        session, clinical_img, dermoscopic_img, metadata)

    # Yedek: bazı sürümler ısı haritasını tahmin cevabının içinde de gönderebilir.
    if gradcam_clinical is None:
        gradcam_clinical = _extract_image(payload, ["clinical_gradcam", "gradcam_clinical"])
    if gradcam_dermoscopic is None:
        gradcam_dermoscopic = _extract_image(payload, ["dermoscopic_gradcam", "gradcam_dermoscopic"])

    return {
        "scores": scores,
        "gradcam_clinical": gradcam_clinical,
        "gradcam_dermoscopic": gradcam_dermoscopic,
    }


def _build_multipart(clinical_img, dermoscopic_img, metadata):
    """İstek gövdesini (görseller + metadata) taze olarak hazırlar."""
    sex_label = metadata.get("sex", "")
    site_label = metadata.get("site", "")
    files = {
        "clinical_image":    ("clinical.jpg",    _to_jpeg_bytes(clinical_img),    "image/jpeg"),
        "dermoscopic_image": ("dermoscopic.jpg", _to_jpeg_bytes(dermoscopic_img), "image/jpeg"),
    }
    data = {
        "age":       str(int(metadata.get("age", 0))),
        "sex":       SEX_API.get(sex_label, str(sex_label).lower()),
        "skin_tone": str(int(metadata.get("skin_tone", 0))),
        "site":      SITE_API.get(site_label, str(site_label)),
    }
    return files, data


def _fetch_gradcam(session, clinical_img, dermoscopic_img, metadata):
    """
    Grad-CAM endpoint'ini çağırır ve iki base64 görseli PIL görsele çevirir.
    Herhangi bir hata olursa (None, None) döner; tahmin akışını bozmaz.
    """
    try:
        files, data = _build_multipart(clinical_img, dermoscopic_img, metadata)
        r = session.post(f"{BACKEND_URL}{GRADCAM_PATH}", files=files, data=data, timeout=TIMEOUT)
        if r.status_code != 200:
            return None, None
        payload = r.json()
        clinical = _extract_image(payload, [
            "clinical_gradcam", "gradcam_clinical", "clinical", "heatmap_clinical"])
        dermoscopic = _extract_image(payload, [
            "dermoscopic_gradcam", "gradcam_dermoscopic", "dermoscopic", "heatmap_dermoscopic"])
        return clinical, dermoscopic
    except Exception:
        return None, None


def _to_jpeg_bytes(img: Image.Image) -> bytes:
    """PIL görselini JPEG bayt dizisine çevirir (istek gövdesinde göndermek için)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return buf.read()


def _extract_image(payload: dict, keys) -> Image.Image | None:
    """
    Cevaptaki olası bir görsel alanını (base64 / data-URL) PIL görsele çevirir.
    Bulamazsa None döndürür (arayüz Grad-CAM bölümünü gizler).
    """
    for k in keys:
        val = payload.get(k)
        if not val or not isinstance(val, str):
            continue
        try:
            # "data:image/png;base64,...." biçimini de destekle
            if "," in val and val.strip().lower().startswith("data:"):
                val = val.split(",", 1)[1]
            raw = base64.b64decode(val)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# MOCK (yedek) — sadece USE_BACKEND = False iken kullanılır.
# ---------------------------------------------------------------------------
def _predict_mock(clinical_img, dermoscopic_img, metadata) -> dict:
    seed = _seed_from_inputs(clinical_img, metadata)
    rng = np.random.default_rng(seed)
    scores = rng.random(len(CLASSES)) * 0.35
    dominant = rng.integers(0, len(CLASSES))
    scores[dominant] = 0.70 + rng.random() * 0.28
    secondary = rng.integers(0, len(CLASSES))
    scores[secondary] = max(scores[secondary], 0.40 + rng.random() * 0.25)
    scores = [float(min(0.99, s)) for s in scores]
    return {
        "scores": scores,
        "gradcam_clinical": _fake_heatmap(clinical_img, rng),
        "gradcam_dermoscopic": _fake_heatmap(dermoscopic_img, rng),
    }


def _seed_from_inputs(img: Image.Image, metadata: dict) -> int:
    small = np.asarray(img.convert("L").resize((16, 16)), dtype=np.int64)
    base = int(small.sum()) + int(metadata.get("age", 0)) * 7
    return base % (2**31)


def _fake_heatmap(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    from matplotlib import colormaps

    base = img.convert("RGB").resize((300, 300))
    arr = np.asarray(base, dtype=np.float32) / 255.0
    h, w = 300, 300
    cy, cx = rng.integers(90, 210), rng.integers(90, 210)
    yy, xx = np.mgrid[0:h, 0:w]
    sigma = rng.integers(45, 80)
    heat = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2)))
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    colored = colormaps["jet"](heat)[:, :, :3]
    alpha = 0.45 * heat[:, :, None]
    blended = (1 - alpha) * arr + alpha * colored
    blended = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(blended)
