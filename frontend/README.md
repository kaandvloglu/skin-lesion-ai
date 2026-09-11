# Cilt Lezyonu Analizi — Frontend (Arayüz)

Bu klasör, projenin **arayüz (frontend)** kısmıdır. Kullanıcı iki fotoğraf
yükler, hasta bilgilerini girer ve 11 kategorilik olasılık sonuçlarını +
ısı haritalarını görür.

> ⚠️ Bu bir tıbbi teşhis aracı değildir. Araştırma/eğitim amaçlı prototiptir.

## Dosyalar
- `app.py` — kullanıcının gördüğü ekran (Streamlit arayüzü)
- `model_service.py` — yapay zeka bağlantısı. Şu an SAHTE (demo) sonuç üretir;
  gerçek model gelince yalnızca bu dosya değişir.
- `requirements.txt` — gerekli Python paketleri

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

## Gerçek modele geçiş
Model / API hazır olduğunda `model_service.py` içindeki `predict()` fonksiyonunun
içi gerçek modele bağlanacak şekilde değiştirilir. `app.py`'ye dokunulmaz.
