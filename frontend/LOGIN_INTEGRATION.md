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

## 4. Kayıt (Register) — ek bilgiler

Giriş ekranına bir **"Sign up" (Kayıt ol)** sekmesi de eklendi (şu an mock).
Gerçek backend'e bağlamak için kayıt tarafı için de şunları öğrenmemiz gerekiyor:

1. **Kayıt adresi ve yöntemi:** ör. `POST https://.../register`
2. **İstenen alanlar:** sadece kullanıcı adı + şifre mi, yoksa e-posta / ad-soyad da var mı?
3. **İstek formatı:** ör. `{ "username": "...", "password": "..." }`
4. **Başarılı cevap:** kayıt sonrası kullanıcı otomatik giriş yapmış mı sayılıyor
   (token dönüyor mu), yoksa "kaydoldu, şimdi giriş yap" mı? Hangi alanlar dönüyor?
5. **Hata durumları:** kullanıcı adı zaten alınmışsa / şifre kuralları
   sağlanmazsa hangi kod/mesaj dönüyor?
6. **Şifre kuralları:** en az uzunluk, karakter kuralı vb. var mı? (Arayüzde
   aynı kuralı gösterelim.)

---

Bu cevaplar gelince giriş ve kayıt ekranlarını gerçek backend'e bağlamak,
`auth_service.py` içinde küçük bir değişiklik olacak.
