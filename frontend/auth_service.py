"""
auth_service.py
----------------
Bu dosya, GİRİŞ EKRANI ile arkadaşının BACKEND doğrulama servisi arasındaki köprüdür.

Şu an backend giriş servisi netleşmediği için burada SAHTE (mock) bir doğrulama var.
Böylece giriş ekranı hemen çalışır ve bitmiş görünür.

Backend hazır olduğunda SADECE bu dosyadaki `login()` fonksiyonunun içini değiştireceğiz
(backend'e istek atıp cevabını döndürecek şekilde); arayüze (app.py) dokunmayacağız.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# GEÇİCİ (mock) kullanıcılar. Backend'e bağlanınca bu tamamen kaldırılacak.
# ---------------------------------------------------------------------------
MOCK_USERS = {
    "demo": "skinai2026",
    "doctor": "skinai2026",
}

# Giriş ekranında geçici demo bilgisini göster. Teslim/rapor için görünmesini
# istemezsen bunu False yap (tek satır).
SHOW_DEMO_HINT = True


def login(username: str, password: str) -> dict:
    """
    Kullanıcıyı doğrular.

    ÇIKTI (sözlük):
      { "ok": bool, "token": str | None, "username": str | None, "error": str | None }

    ============================================================
    ŞU AN: Aşağısı SAHTE doğrulama yapar (demo).
    GERÇEK BACKEND'E GEÇERKEN: Bu fonksiyonun içini şu şekilde değiştireceğiz:
        - Kullanıcı adı + şifreyi backend'in giriş adresine (ör. POST /login) gönder
        - Backend "başarılı" derse dönen token'ı al ve {"ok": True, "token": ...} döndür
        - Başarısızsa {"ok": False, "error": "..."} döndür
    app.py hiç değişmeyecek.
    ============================================================
    """
    username = (username or "").strip()
    if username in MOCK_USERS and password == MOCK_USERS[username]:
        return {"ok": True, "token": "mock-token", "username": username, "error": None}
    return {"ok": False, "token": None, "username": None,
            "error": "Invalid username or password. Please try again."}
