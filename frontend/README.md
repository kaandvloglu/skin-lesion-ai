# Cilt Lezyonu Analizi — Frontend (Arayüz)

Bu klasör, projenin **arayüz (frontend)** kısmıdır. Kullanıcı giriş yapar, aynı
lezyonun iki fotoğrafını (klinik + dermoskopik) yükler, hasta bilgilerini girer;
uygulama 11 kategorilik olasılık sonuçlarını, belirsizlik uyarısını ve modelin
nereye baktığını gösteren Grad-CAM ısı haritalarını gösterir.

Arayüz **Python + Streamlit** ile yazılmıştır ve gerçek backend'e (Spring Boot)
bağlıdır.

> ⚠️ Bu bir tıbbi teşhis aracı değildir. Araştırma/eğitim amaçlı bir karar-destek
> prototipidir.

## Canlı uygulama
https://skin-lesion-ai-project.streamlit.app

## Dosyalar
- `app.py` — kullanıcının gördüğü ekran (Streamlit arayüzü): giriş/kayıt,
  görsel yükleme, sonuçlar, Grad-CAM ve sonucu indirme.
- `model_service.py` — tahmin bağlantısı. Görselleri ve hasta bilgilerini
  backend'in tahmin uçlarına gönderir, 11 skoru ve Grad-CAM ısı haritalarını alır.
- `auth_service.py` — giriş/kayıt bağlantısı (e-posta + şifre).
- `requirements.txt` — gerekli Python paketleri.
- `sample_images/` — test için örnek klinik ve dermoskopik görseller.

## Kurulum (tek seferlik)
```bash
pip install -r requirements.txt
```

## Çalıştırma
```bash
python3 -m streamlit run app.py
```
Komuttan sonra tarayıcıda otomatik olarak `http://localhost:8501` açılır.
(Not: `streamlit` komutu doğrudan çalışmazsa yukarıdaki `python3 -m ...` biçimini kullan.)

## Mimari
Arayüz yalnızca backend'e (Spring Boot) HTTP üzerinden konuşur; görselleri
`multipart/form-data` olarak gönderir ve sonucu JSON olarak alır. Tahmin ve
giriş mantığı `model_service.py` ve `auth_service.py` içinde ayrılmıştır; böylece
backend tarafında bir değişiklik olduğunda `app.py`'ye (arayüze) dokunmak
gerekmez.
