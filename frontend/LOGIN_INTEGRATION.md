# Frontend → Backend: Giriş (Login) Entegrasyon İsteği

> Bu dosyayı backend / güvenlik tarafındaki arkadaşına ilet. Frontend'in giriş
> ekranı hazır ve şu an geçici (mock) doğrulama ile çalışıyor. Gerçek backend
> girişine bağlamak için aşağıdaki bilgilere ihtiyacımız var. Bağlantı yapılınca
> yalnızca `auth_service.py` dosyasının içi değişecek; arayüze dokunulmayacak.

---

## 1. Frontend NE gönderecek

Giriş ekranındaki iki alan:

- `username` (metin)
- `password` (metin)

## 2. Frontend NE bekliyor

- Giriş **başarılı mı** bilgisi (başarılı / başarısız).
- Başarılıysa, sonraki isteklerde kimlik için kullanılacak bir **token** (varsa).
- İsteğe bağlı: kullanıcı adı ve/veya rol bilgisi.

Kısaca `auth_service.login()` şu formatta bir sonuç döndürmeli:

```json
{ "ok": true, "token": "<token>", "username": "<ad>", "error": null }
```
başarısızsa:
```json
{ "ok": false, "token": null, "username": null, "error": "Invalid username or password." }
```

## 3. Backend'den öğrenmemiz gerekenler (lütfen cevapla)

1. **Adres ve yöntem:** Giriş isteğini nereye atacağız? (ör. `POST https://.../login`)
2. **İstek formatı:** JSON mı? Gövde nasıl? (ör. `{ "username": "...", "password": "..." }`)
3. **Başarılı cevap:** Ne dönüyor? Token var mı, JSON'da hangi alanda? (ör. `{ "access_token": "..." }`)
4. **Başarısız cevap:** Hangi durum kodu / mesaj dönüyor? (ör. `401` + `{ "detail": "..." }`)
5. **Token kullanımı:** Token'ı sonraki isteklerde (özellikle model tahmini isteğinde)
   nasıl göndereceğiz? (ör. `Authorization: Bearer <token>` başlığı)
6. **Kullanıcılar:** Hesaplar nasıl oluşturuluyor? Sabit test hesapları var mı?
   Kayıt (register) ekranı gerekiyor mu, yoksa sadece giriş mi?
7. **Erişim/CORS:** Uygulama Streamlit Community Cloud'dan (internetten) backend'e
   erişecek. Backend'in dışarıdan erişime (HTTPS/CORS) açık olması gerekiyor —
   bu tarafta bir ayar gerekiyor mu?

---

Bu cevaplar gelince giriş ekranını gerçek backend'e bağlamak, `auth_service.py`
içinde küçük bir değişiklik olacak.
