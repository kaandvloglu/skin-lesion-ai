"""
app.py — Skin Lesion Analysis | Decision-Support Interface (Frontend)
=====================================================================
Kullanıcının gördüğü EKRAN. Çalıştırmak için terminalde:

    python3 -m streamlit run app.py

Arayüz metinleri İngilizce; kod yorumları (senin anlaman için) Türkçe.
Tahmin `model_service.py`'den gelir; şu an sonuçlar SAHTE (demo) veridir.

HCI notu: Nielsen'in 10 kullanılabilirlik prensibi gözetildi
(sistem durumu görünürlüğü, hata önleme, hatadan kurtarma, kullanıcı
kontrolü/özgürlüğü, yardım & dokümantasyon, tutarlılık, minimalizm).
"""

import time
from datetime import datetime

import streamlit as st
from PIL import Image, UnidentifiedImageError

import model_service as ms

# ---------------------------------------------------------------------------
# Sabitler / ayarlar
# ---------------------------------------------------------------------------
MAX_FILE_MB = 10  # tek dosya için üst sınır (error prevention)

st.set_page_config(
    page_title="Skin Lesion Analysis — Decision Support",
    page_icon="🔬",
    layout="wide",
)

# Klinik gruplara göre renkler (yüksek kontrast; renk TEK BAŞINA anlam taşımaz,
# her çubukta metin etiketi de vardır — erişilebilirlik / renk körlüğü).
GROUP_COLORS = {
    "Benign":        "#1f8a4c",  # green
    "Inflammatory":  "#2f6fd0",  # blue
    "Pre-malignant": "#c07a00",  # amber
    "Malignant":     "#c02626",  # red
}

# Grup -> düz dille özet cümlesi (Adım 3: iyi/kötü huylu özet bandı)
GROUP_SUMMARY = {
    "Benign":        "The most likely category is benign.",
    "Inflammatory":  "The most likely category is an inflammatory or infectious condition.",
    "Pre-malignant": "The most likely category is pre-malignant (early or in-situ).",
    "Malignant":     "The most likely category is a malignant type.",
}

# ---------------------------------------------------------------------------
# Oturum durumu (session state) — sonucu saklamak ve "Clear" için
# ---------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "result_meta" not in st.session_state:
    st.session_state.result_meta = None


def clear_all():
    """Reset / New analysis — kullanıcı kontrolü & özgürlüğü (Nielsen #3)."""
    st.session_state.result = None
    st.session_state.result_meta = None
    for k in ("clinical", "dermo"):
        st.session_state.pop(k, None)


# ---------------------------------------------------------------------------
# Başlık + KALICI TIBBİ UYARI (non-diagnostic, Nielsen #10 help & docs)
# ---------------------------------------------------------------------------
st.title("🔬 Skin Lesion Analysis — Decision Support")
st.caption("Multimodal decision-support prototype based on the MILK10k dataset")

st.warning(
    "**This is not a diagnosis. Consult a specialist.**  This tool is a "
    "research and educational decision-support prototype only. It does not "
    "replace a medical examination. Always seek a qualified doctor for any "
    "health decision."
)

# Yardım / dokümantasyon (Nielsen #10) — istenirse açılır, ekranı kalabalıklaştırmaz
with st.expander("ℹ️ How to use this tool"):
    st.markdown(
        """
        1. **Upload two photos of the same lesion:** a *clinical* close-up and
           a *dermoscopic* image (JPG or PNG, up to {mb} MB each).
        2. **Enter the patient details:** age, sex, skin tone and body site.
        3. Click **Analyze**. Within a few seconds you will see:
           - the **likelihood for each of the 11 categories** (each score is
             independent, so they do **not** add up to 100%),
           - two **Grad-CAM heatmaps** showing where the model looked,
        4. Use **Clear** to start a new analysis at any time.
        """.format(mb=MAX_FILE_MB)
    )

st.divider()


# ---------------------------------------------------------------------------
# Yardımcı: yüklenen dosyayı doğrula ve görsele çevir
# (error prevention + recover from errors, Nielsen #5 & #9)
# ---------------------------------------------------------------------------
def load_valid_image(uploaded, label: str):
    """
    (image, error_message) döndürür. Hata varsa image=None olur ve
    kullanıcının NE YAPMASI GEREKTİĞİNİ söyleyen bir mesaj döner.
    """
    if uploaded is None:
        return None, None
    size_mb = uploaded.size / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        return None, (
            f"**{label}** is too large ({size_mb:.1f} MB). "
            f"Please upload an image smaller than {MAX_FILE_MB} MB."
        )
    try:
        img = Image.open(uploaded)
        img.verify()              # bozuk dosya kontrolü
        uploaded.seek(0)          # verify sonrası başa sar
        img = Image.open(uploaded)
        return img, None
    except (UnidentifiedImageError, OSError):
        return None, (
            f"**{label}** could not be read. Please make sure it is a valid "
            "JPG or PNG image and try again."
        )


# ---------------------------------------------------------------------------
# 1) GİRDİ BÖLÜMÜ
# ---------------------------------------------------------------------------
st.subheader("1) Provide the images and patient details")

col_img1, col_img2, col_meta = st.columns([1, 1, 1])

with col_img1:
    st.markdown("**Clinical photo**")
    clinical_file = st.file_uploader(
        "Close-up photo of the lesion",
        type=["jpg", "jpeg", "png"],
        key="clinical",
        help="A normal (non-dermoscopic) close-up photograph of the lesion.",
    )
    clinical_img, clinical_err = load_valid_image(clinical_file, "Clinical photo")
    if clinical_err:
        st.error(clinical_err)
    elif clinical_img is not None:
        st.image(clinical_img, use_container_width=True)

with col_img2:
    st.markdown("**Dermoscopic photo**")
    dermo_file = st.file_uploader(
        "Photo taken with a dermoscope",
        type=["jpg", "jpeg", "png"],
        key="dermo",
        help="An image of the same lesion captured through a dermoscope.",
    )
    dermo_img, dermo_err = load_valid_image(dermo_file, "Dermoscopic photo")
    if dermo_err:
        st.error(dermo_err)
    elif dermo_img is not None:
        st.image(dermo_img, use_container_width=True)

with col_meta:
    st.markdown("**Patient details**")
    age = st.slider("Age", min_value=0, max_value=100, value=45, step=5,
                    help="Approximate age of the patient.")
    sex = st.radio("Sex", ms.SEX_OPTIONS, horizontal=True)
    skin_tone = st.slider(
        "Skin tone (0 = lightest, 5 = darkest)",
        min_value=0, max_value=5, value=2,
        help="MILK10k skin-tone scale from 0 (lightest) to 5 (darkest).",
    )
    site = st.selectbox("Body site", ms.ANATOM_SITES, index=1,
                        help="Where on the body the lesion is located.")

# Girişler tam mı? (error prevention — buton eksik girişte pasif kalır)
both_images_ready = (
    clinical_img is not None and clinical_err is None
    and dermo_img is not None and dermo_err is None
)

st.write("")
c_analyze, c_clear = st.columns([3, 1])
with c_analyze:
    analyze = st.button(
        "🔎 Analyze",
        type="primary",
        use_container_width=True,
        disabled=not both_images_ready,   # Nielsen #5: hata oluşmadan önle
    )
with c_clear:
    st.button("↺ Clear", use_container_width=True, on_click=clear_all)

# Butonun neden pasif olduğunu açıkça söyle (sistem durumu görünürlüğü)
if not both_images_ready:
    st.caption("⤷ Upload **both** a clinical and a dermoscopic photo to enable analysis.")

st.divider()


# ---------------------------------------------------------------------------
# Yardımcı: tek bir olasılık çubuğu (renk + metin, yüksek kontrast)
# ---------------------------------------------------------------------------
def render_bar(code: str, name: str, pct: float, color: str):
    st.markdown(
        f"""
        <div style="margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;
                      font-size:0.9rem;margin-bottom:3px;">
            <span><b>{code}</b> <span style="color:#666;">— {name}</span></span>
            <span style="font-variant-numeric:tabular-nums;"><b>{pct:.0f}%</b></span>
          </div>
          <div style="background:#e6e6e6;border-radius:6px;height:16px;width:100%;">
            <div style="background:{color};width:{pct:.0f}%;height:16px;
                        border-radius:6px;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# ANALİZ: butona basınca, aşamalı durum göstergesiyle (Nielsen #1)
# ---------------------------------------------------------------------------
if analyze and both_images_ready:
    metadata = {"age": age, "sex": sex, "skin_tone": skin_tone, "site": site}
    try:
        with st.status("Analyzing…", expanded=True) as status:
            st.write("Reading images…")
            time.sleep(0.4)
            st.write("Running the model…")
            result = ms.predict(clinical_img, dermo_img, metadata)
            st.write("Generating Grad-CAM heatmaps…")
            time.sleep(0.4)
            status.update(label="Analysis complete", state="complete", expanded=False)
        st.session_state.result = result
        st.session_state.result_meta = metadata
    except Exception:
        # Beklenmedik hata — kullanıcıya ne yapacağını söyle (Nielsen #9)
        st.session_state.result = None
        st.error(
            "Something went wrong while analyzing the images. Please click "
            "**Clear**, check your photos, and try again."
        )


# ---------------------------------------------------------------------------
# 2) SONUÇLAR (session_state'te varsa göster — Clear'a kadar kalır)
# ---------------------------------------------------------------------------
result = st.session_state.result
if result is not None:
    scores = result["scores"]
    ranked = sorted(zip(ms.CLASSES, scores), key=lambda x: x[1], reverse=True)

    st.subheader("2) Results")

    top_class, top_score = ranked[0]
    second_class, second_score = ranked[1]
    top_color = GROUP_COLORS.get(top_class["group"], "#555")

    # --- Özet bandı: iyi/kötü huylu grubu düz dille (Adım 3) ----------------
    summary_text = GROUP_SUMMARY.get(top_class["group"], "See the category scores below.")
    st.markdown(
        f"""
        <div style="background:{top_color};color:#fff;border-radius:10px;
                    padding:14px 18px;margin-bottom:12px;">
          <div style="font-size:1.15rem;font-weight:600;">{summary_text}</div>
          <div style="font-size:0.9rem;opacity:0.92;margin-top:2px;">
               This is not a diagnosis — it only reflects the model's scores. Consult a specialist.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- En olası tahmin — vurgulu kart ------------------------------------
    st.markdown(
        f"""
        <div style="border:2px solid {top_color};border-radius:10px;
                    padding:14px 18px;margin-bottom:6px;">
          <div style="font-size:0.8rem;color:#666;letter-spacing:0.04em;">
               MOST LIKELY CATEGORY</div>
          <div style="font-size:1.4rem;"><b>{top_class['name']}</b>
               <span style="color:#666;font-size:1rem;">({top_class['code']})</span></div>
          <div style="font-size:1rem;color:{top_color};">
               {top_class['group']} — {top_score*100:.0f}% likelihood</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Belirsizlik notu (Adım 3): iki üst skor yakınsa veya en yüksek düşükse
    gap = top_score - second_score
    if gap < 0.15:
        st.warning(
            f"⚖️ **The result is uncertain.** The top two categories have similar "
            f"scores ({top_class['code']} {top_score*100:.0f}% vs "
            f"{second_class['code']} {second_score*100:.0f}%). Interpret with extra "
            "caution and rely on a specialist."
        )
    elif top_score < 0.50:
        st.warning(
            "⚖️ **Low confidence.** Even the highest score is low, so the model is "
            "not confident about this lesion. Interpret with extra caution."
        )

    st.caption(
        f"⏱️ Inference time: {result['inference_seconds']:.2f} s · "
        "Each category has its own independent score, so the values do not add "
        "up to 100%. All 11 categories are listed below, highest first."
    )

    st.write("")
    for cls, s in ranked:
        color = GROUP_COLORS.get(cls["group"], "#888")
        render_bar(cls["code"], cls["name"], s * 100, color)

    # Renk açıklaması (lejant) — renk + metin birlikte
    legend = "&nbsp;&nbsp;".join(
        f"<span style='color:{c};font-size:1.1rem;'>■</span> {g}"
        for g, c in GROUP_COLORS.items()
    )
    st.markdown(
        f"<div style='margin-top:10px;font-size:0.85rem;'>{legend}</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Grad-CAM ısı haritaları
    st.subheader("3) Where the model looked (Grad-CAM)")
    st.caption(
        "The red-yellow areas are where the model focused most when making its "
        "prediction. A separate heatmap is shown for each image."
    )
    gc1, gc2 = st.columns(2)
    with gc1:
        st.markdown("**Clinical photo**")
        st.image(result["gradcam_clinical"], use_container_width=True)
    with gc2:
        st.markdown("**Dermoscopic photo**")
        st.image(result["gradcam_dermoscopic"], use_container_width=True)

    st.divider()

    # --- İndirilebilir sonuç özeti (Adım 3) --------------------------------
    st.subheader("4) Save the result")
    meta = st.session_state.result_meta or {}
    lines = [
        "SKIN LESION ANALYSIS — RESULT SUMMARY",
        "(Research/education prototype. This is not a diagnosis. Consult a specialist.)",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Patient details:",
        f"  Age:        {meta.get('age', '-')}",
        f"  Sex:        {meta.get('sex', '-')}",
        f"  Skin tone:  {meta.get('skin_tone', '-')} (0=lightest, 5=darkest)",
        f"  Body site:  {meta.get('site', '-')}",
        "",
        f"Most likely category: {top_class['name']} ({top_class['code']}) — "
        f"{top_class['group']}, {top_score*100:.0f}% likelihood",
        "",
        "All categories (independent scores, do not sum to 100%):",
    ]
    for cls, s in ranked:
        lines.append(f"  {s*100:5.0f}%  {cls['code']:<8} {cls['name']} [{cls['group']}]")
    summary_txt = "\n".join(lines)

    st.download_button(
        "⬇️ Download result summary (.txt)",
        data=summary_txt,
        file_name=f"skin_lesion_result_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        help="Save a text file with the inputs, all 11 scores and the disclaimer.",
    )

    st.divider()
    st.info(
        "ℹ️ These results are currently **demo (placeholder) data**. Once the AI "
        "model is connected, real predictions will appear here automatically — "
        "no changes to the interface are needed."
    )

elif not analyze:
    # Henüz sonuç yok — ne yapılacağını söyle (sistem durumu görünürlüğü)
    st.info("👆 Upload the two photos and patient details, then click **Analyze**.")
