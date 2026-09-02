# Frontend → AI/Backend: Entegrasyon İsteği (API Sözleşmesi)

> Bu dosyayı AI/model tarafındaki arkadaşına ilet. Amaç: frontend'in modele
> sorunsuz bağlanabilmesi için iki tarafın **aynı girdi/çıktı formatında**
> anlaşması. Frontend hazır; sadece modelin bu formatta sonuç vermesi gerekiyor.

---

## 1. Frontend modele NE gönderecek (girdiler)

Her analizde tek bir lezyon için şunlar:

- **clinical_image** — klinik (normal) yakın çekim foto (JPG/PNG)
- **dermoscopic_image** — dermoskopik foto (JPG/PNG)
- **metadata**:
  - `age`: tam sayı (ör. 45)
  - `sex`: `"Female"` | `"Male"` | `"Unspecified"`
  - `skin_tone`: 0–5 arası tam sayı (MILK10k skalası)
  - `site`: metin — şu değerlerden biri:
    `"Head / neck"`, `"Anterior torso"`, `"Posterior torso"`,
    `"Upper limb (arm)"`, `"Lower limb (leg)"`, `"Palms / soles"`,
    `"Oral / genital"`, `"Unknown"`

> Not: Görselleri modelin beklediği boyuta (ör. 300×300) **model tarafı** mı
> yeniden boyutlandıracak, yoksa frontend mi göndersin? Lütfen belirt.

## 2. Frontend modelden NE bekliyor (çıktılar)

Tek bir sonuç nesnesi:

- **scores** — 11 kategori için **bağımsız** olasılık (her biri 0.0–1.0 arası,
  sigmoid; toplamları 1 OLMAK ZORUNDA DEĞİL). Karışıklık olmasın diye lütfen
  **sınıf koduna göre bir sözlük (dict)** olarak döndür:

  ```json
  {
    "NV": 0.03, "BKL": 0.12, "DF": 0.01, "VASC": 0.35, "BEN_OTH": 0.35,
    "INF": 0.76, "AKIEC": 0.33, "BCC": 0.12, "SCCKA": 0.62,
    "MEL": 0.08, "MAL_OTH": 0.01
  }
  ```

  (11 kod tam olarak bunlar: NV, BKL, DF, VASC, BEN_OTH, INF, AKIEC, BCC,
  SCCKA, MEL, MAL_OTH)

- **gradcam_clinical** — klinik görsel için Grad-CAM ısı haritası
- **gradcam_dermoscopic** — dermoskopik görsel için Grad-CAM ısı haritası
  (ısı haritaları orijinal görsele bindirilmiş PNG olarak dönebilir)

- **(opsiyonel) thresholds** — her sınıf için karar eşiği (hepsi 0.5 değilse).
  Varsa frontend gösterebilir; yoksa sorun değil.

## 3. Nasıl çağıracağız? (İki seçenek — biri yeterli)

**Seçenek A — Python fonksiyonu (en basit):**
Bize import edilebilir bir fonksiyon ver, imzası şöyle olsun:

```python
def predict(clinical_img, dermoscopic_img, metadata: dict) -> dict:
    # döndürür: {"scores": {...}, "gradcam_clinical": <PIL.Image>,
    #            "gradcam_dermoscopic": <PIL.Image>}
```

**Seçenek B — Web API (FastAPI vb.):**
Bir URL ver (ör. `POST /predict`), iki görsel + metadata alsın, yukarıdaki
JSON'u (+ ısı haritalarını base64 veya URL olarak) döndürsün. Örnek yanıt:

```json
{
  "scores": { "...": 0.0 },
  "gradcam_clinical": "data:image/png;base64,...",
  "gradcam_dermoscopic": "data:image/png;base64,..."
}
```

## 4. Diğer sorular (lütfen cevapla)

1. Seçenek A mı B mi? (Python fonksiyonu mu, web API mi?)
2. Görselleri kim yeniden boyutlandıracak (frontend mi, model mi)?
3. Grad-CAM'i sen mi üreteceksin (evet bekliyoruz), yoksa sadece skor mu döner?
4. Ortalama yanıt süresi ~10 sn altında kalabilir mi? (Capstone hedefi)
5. Model hata verirse (ör. görsel bozuk) nasıl haber verirsin (hata mesajı formatı)?

---

Bu 5 soruya ve yukarıdaki formata karar verdiğimizde, frontend'i bağlamak
sadece `model_service.py` içinde küçük bir değişiklik olacak.
