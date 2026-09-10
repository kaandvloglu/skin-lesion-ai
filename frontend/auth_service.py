"""
auth_service.py
----------------
Bu dosya, GİRİŞ / KAYIT EKRANI ile arkadaşının BACKEND doğrulama servisi
(Spring Boot) arasındaki köprüdür.

Backend artık internette (public HTTPS) yayında, o yüzden burada GERÇEK
backend'e bağlanıyoruz. Backend giriş/kayıt için e-posta kullanıyor ve oturumu
çerez (cookie / Spring Session) ile tutuyor; o yüzden aynı `requests.Session`
nesnesini kullanıyoruz ki giriş çerezimiz sonraki isteklerde de taşınsın.

GÜVENLİK ANAHTARI:
    Aşağıdaki USE_BACKEND = True iken gerçek backend'e bağlanır.
    Bir şey ters giderse tek satırı `USE_BACKEND = False` yap; uygulama anında
    eski çalışan SAHTE (mock) haline döner. Böylece canlı uygulama asla bozulmaz.

app.py'ye (arayüze) dokunulmadan giriş çalışır; sadece alan adları (username
yerine email) app.py'de güncellendi.
"""

from __future__ import annotations

import requests

# ---------------------------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------------------------
# Gerçek backend'e mi bağlanılsın? (Sorun olursa False yap → mock'a döner.)
USE_BACKEND = True

# Arkadaşının verdiği public backend adresi (sonunda / OLMADAN).
BACKEND_URL = "https://skin-lesion-backend-nnfv.onrender.com"

# İstek zaman aşımı (saniye). Render ücretsiz plan ilk istekte "uykudan"
# uyanabildiği için ilk giriş 30–60 sn sürebilir; bu yüzden geniş tutuldu.
TIMEOUT = 120

# Kayıt için en az şifre uzunluğu (arayüzde göstermek için; asıl kuralı
# backend belirler, backend farklı bir hata dönerse onu gösteririz).
MIN_PASSWORD_LEN = 6

# Giriş ekranında demo bilgisi gösterme (gerçek backend'de demo hesap yok).
SHOW_DEMO_HINT = False

# ---------------------------------------------------------------------------
# Oturum (session) — giriş çerezini tutar ve tahmin isteğinde de kullanılır.
# model_service.py bu oturumu `get_session()` ile alır, böylece giriş çerezi
# tahmin isteğine de taşınır.
# ---------------------------------------------------------------------------
_session = requests.Session()


def get_session() -> requests.Session:
    """Giriş çerezini taşıyan ortak oturum nesnesini döndürür."""
    return _session


# ---------------------------------------------------------------------------
# GEÇİCİ (mock) kullanıcılar — sadece USE_BACKEND = False iken kullanılır.
# ---------------------------------------------------------------------------
_MOCK_USERS = {
    "demo@skinai.local": "skinai2026",
    "doctor@skinai.local": "skinai2026",
}


def _friendly_network_error() -> dict:
    """Ağ/bağlantı hatası için kullanıcı dostu ortak mesaj."""
    return {
        "ok": False, "token": None, "username": None,
        "error": ("Could not reach the server. It may be starting up — this can "
                  "take up to a minute on the first try. Please wait a moment and "
                  "try again."),
    }


def login(email: str, password: str) -> dict:
    """
    Kullanıcıyı doğrular.

    ÇIKTI (sözlük):
      { "ok": bool, "token": str | None, "username": str | None, "error": str | None }
    """
    email = (email or "").strip()

    # ----- GERÇEK BACKEND -----
    if USE_BACKEND:
        if not email or not password:
            return {"ok": False, "token": None, "username": None,
                    "error": "Please enter your email and password."}
        try:
            r = _session.post(
                f"{BACKEND_URL}/api/auth/login",
                json={"email": email, "password": password},
                timeout=TIMEOUT,
            )
        except requests.RequestException:
            return _friendly_network_error()

        if r.status_code == 200:
            data = _safe_json(r)
            name = data.get("name") or data.get("username") or email
            token = data.get("token") or data.get("accessToken") or "session"
            return {"ok": True, "token": token, "username": name, "error": None}
        if r.status_code in (400, 401, 403):
            return {"ok": False, "token": None, "username": None,
                    "error": "Invalid email or password. Please try again."}
        return {"ok": False, "token": None, "username": None,
                "error": f"Login failed (server error {r.status_code}). Please try again."}

    # ----- MOCK (yedek) -----
    if email in _MOCK_USERS and password == _MOCK_USERS[email]:
        return {"ok": True, "token": "mock-token", "username": email, "error": None}
    return {"ok": False, "token": None, "username": None,
            "error": "Invalid email or password. Please try again."}


def register(name: str, email: str, password: str) -> dict:
    """
    Yeni kullanıcı kaydı yapar.

    ÇIKTI: login() ile AYNI format.
    Not: Backend kayıt sonrası otomatik giriş yaptırmıyorsa, arayüz kullanıcıyı
    yine de içeri alır (kayıt başarılıysa). Gerekirse ileride "kaydoldun, şimdi
    giriş yap" akışına çevrilebilir.
    """
    name = (name or "").strip()
    email = (email or "").strip()

    # ----- GERÇEK BACKEND -----
    if USE_BACKEND:
        if not name:
            return {"ok": False, "token": None, "username": None,
                    "error": "Please enter your name."}
        if not email:
            return {"ok": False, "token": None, "username": None,
                    "error": "Please enter your email."}
        if len(password) < MIN_PASSWORD_LEN:
            return {"ok": False, "token": None, "username": None,
                    "error": f"Password must be at least {MIN_PASSWORD_LEN} characters."}
        try:
            r = _session.post(
                f"{BACKEND_URL}/api/auth/register",
                json={"name": name, "email": email, "password": password},
                timeout=TIMEOUT,
            )
        except requests.RequestException:
            return _friendly_network_error()

        if r.status_code in (200, 201):
            return {"ok": True, "token": "session", "username": name, "error": None}
        if r.status_code == 409:
            return {"ok": False, "token": None, "username": None,
                    "error": "This email is already registered. Please sign in instead."}
        if r.status_code == 400:
            # Backend'in kendi doğrulama mesajı varsa onu göster
            data = _safe_json(r)
            msg = data.get("message") or data.get("error") \
                or "Please check your details and try again."
            return {"ok": False, "token": None, "username": None, "error": msg}
        return {"ok": False, "token": None, "username": None,
                "error": f"Registration failed (server error {r.status_code}). Please try again."}

    # ----- MOCK (yedek) -----
    if not name:
        return {"ok": False, "token": None, "username": None,
                "error": "Please enter your name."}
    if not email:
        return {"ok": False, "token": None, "username": None,
                "error": "Please enter your email."}
    if email in _MOCK_USERS:
        return {"ok": False, "token": None, "username": None,
                "error": "This email is already registered. Please sign in instead."}
    if len(password) < MIN_PASSWORD_LEN:
        return {"ok": False, "token": None, "username": None,
                "error": f"Password must be at least {MIN_PASSWORD_LEN} characters."}
    _MOCK_USERS[email] = password
    return {"ok": True, "token": "mock-token", "username": name, "error": None}


def _safe_json(r: requests.Response) -> dict:
    """Cevap JSON değilse boş sözlük döndür (çökme olmasın)."""
    try:
        data = r.json()
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}
