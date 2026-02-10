import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image, ImageOps

# --- APP KONFIGURATION ---
st.set_page_config(page_title="Präzisions-Analyse Pro", layout="centered")

st.title("🛠 Profil-Mess-App Pro")

# --- SEITENLEISTE (EINSTELLUNGEN) ---
st.sidebar.header("Einstellungen")

orientierung = st.sidebar.radio(
    "Wie liegt das Bauteil im Bild?",
    ("Horizontal (Liegend)", "Vertikal (Stehend)"),
    help="Wähle 'Horizontal', wenn das Bauteil quer im Bild liegt."
)

# Schwellenwerte für die Kantenerkennung
st.sidebar.subheader("Analyse-Parameter")
kanten_sens = st.sidebar.slider(
    "Kanten-Sensibilität", 
    min_value=0.01, max_value=0.50, value=0.14, step=0.01,
    help="Ab welcher Steilheit im Diagramm eine Kante erkannt wird."
)

ref_weiss_mm = st.sidebar.number_input("Referenzbreite Innen (mm)", value=90.00)
such_offset_px_val = st.sidebar.slider("Such-Versatz (Pixel)", 1, 100, 30)
mm_pro_drehung = st.sidebar.number_input("mm pro Umdrehung", value=0.75)

# --- BILD-EINGABE ---
input_method = st.radio("Bildquelle:", ("Kamera nutzen", "Screenshot / Datei hochladen"))

uploaded_file = None
if input_method == "Kamera nutzen":
    uploaded_file = st.camera_input("Foto aufnehmen")
else:
    uploaded_file = st.file_uploader("Bild auswählen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Bild laden & Vorbereiten
    pil_img = Image.open(uploaded_file)
    pil_img = ImageOps.exif_transpose(pil_img)
    
    if pil_img.width > 2000:
        ratio = 2000 / float(pil_img.width)
        new_h = int(float(pil_img.height) * float(ratio))
        pil_img = pil_img.resize((2000, new_h), Image.Resampling.LANCZOS)
    
    img_rgb = np.array(pil_img.convert('RGB'))
    
    if orientierung == "Horizontal (Liegend)":
        img_rot = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)
    else:
        img_rot = img_rgb

    # 2. Graustufen & Gradienten-Profil
    gray = (0.2989 * img_rot[:,:,0] + 0.5870 * img_rot[:,:,1] + 0.1140 * img_rot[:,:,2]).astype(np.float64)
    kernel = np.ones((5, 5), np.float64) / 25.0
    gray_smooth = convolve2d(gray, kernel, mode='same')
    
    h_grad = np.abs(np.diff(gray_smooth, axis=1))
    h_grad = np.hstack((h_grad, np.zeros((h_grad.shape[0], 1))))
    kanten_profil = np.mean(h_grad, axis=0)
    kanten_profil = kanten_profil / np.max(kanten_profil)

    # --- NEUE LOGIK: KANTENSUCHE VON DER MITTE ---
    w_img = kanten_profil.shape[0]
    img_center = w_img // 2

    # 1 & 2: Innenkanten suchen (Grün)
    # Rechts von Mitte suchen
    suche_rechts_w = np.where(kanten_profil[img_center:] > kanten_sens)[0]
    x_rechts_w_px = (img_center + suche_rechts_w[0]) if len(suche_rechts_w) > 0 else img_center
    
    # Links von Mitte suchen (Array umdrehen für Suche nach links)
    suche_links_w = np.where(kanten_profil[:img_center][::-1] > kanten_sens)[0]
    x_links_w_px = (img_center - suche_links_w[0]) if len(suche_links_w) > 0 else img_center

    if x_rechts_w_px > x_links_w_px:
        # Maßstab berechnen
        px_pro_mm = (x_rechts_w_px - x_links_w_px) / ref_weiss_mm
        mm_per_px = 1.0 / px_pro_mm

        # 3: Rechte Außenkante (Gelb)
        # Start: Innenkante + Versatz, dann Suche nach LINKS
        start_r_aussen = min(w_img - 1, x_rechts_w_px + such_offset_px_val)
        suche_r_aussen_bereich = kanten_profil[x_rechts_w_px + 5 : start_r_aussen + 1]
        idx_r_a = np.where(suche_r_aussen_bereich[::-1] > kanten_sens)[0]
        x_rechts_a_px = (start_r_aussen - idx_r_a[0]) if len(idx_r_a) > 0 else start_r_aussen

        # 4: Linke Außenkante (Gelb)
        # Start: Innenkante - Versatz, dann Suche nach RECHTS
        start_l_aussen = max(0, x_links_w_px - such_offset_px_val)
        suche_l_aussen_bereich = kanten_profil[start_l_aussen : x_links_w_px - 5]
        idx_l_a = np.where(suche_l_aussen_bereich > kanten_sens)[0]
        x_links_a_px = (start_l_aussen + idx_l_a[0]) if len(idx_l_a) > 0 else start_l_aussen

        # Berechnung der Korrektur
        zentrum_ist_px = (x_links_w_px + x_rechts_w_px) / 2.0
        zentrum_soll_px = (x_links_a_px + x_rechts_a_px) / 2.0
        abweichung_mm = (zentrum_ist_px - zentrum_soll_px) * mm_per_px
        umdrehungen = abs(abweichung_mm) / mm_pro_drehung
        anweisung = "RECHTS herum" if abweichung_mm <= 0 else "LINKS herum"

        # --- VISUALISIERUNG & METRIKEN ---
        col1, col2 = st.columns(2)
        col1.metric("Abweichung", f"{abs(abweichung_mm):.2f} mm")
        col2.metric("Korrektur", f"{umdrehungen:.2f} Umdr.")
        st.success(f"Drehe die Schraube **{anweisung}**.")

        img_marked = img_rot.copy()
        h_img = img_marked.shape[0]
        # Gelb: Außenkanten
        cv2.line(img_marked, (int(x_links_a_px), 0), (int(x_links_a_px), h_img), (255, 255, 0), 8)
        cv2.line(img_marked, (int(x_rechts_a_px), 0), (int(x_rechts_a_px), h_img), (255, 255, 0), 8)
        # Grün: Innenkanten
        cv2.line(img_marked, (int(x_links_w_px), 0), (int(x_links_w_px), h_img), (0, 255, 0), 8)
        cv2.line(img_marked, (int(x_rechts_w_px), 0), (int(x_rechts_w_px), h_img), (0, 255, 0), 8)
        # Linien für Zentrum
        cv2.line(img_marked, (int(zentrum_soll_px), 0), (int(zentrum_soll_px), h_img), (0, 0, 255), 4) # Blau
        cv2.line(img_marked, (int(zentrum_ist_px), 0), (int(zentrum_ist_px), h_img), (255, 0, 0), 4) # Rot

        st.subheader("🔍 Detail-Zoom")
        z_cols = st.columns(2)
        y_mid = h_img // 2 
        def get_zoom(img, x, y, size=100, scale=5):
            x1, y1 = max(0, int(x - size//2)), max(0, int(y - size//2))
            x2, y2 = min(img.shape[1], int(x + size//2)), min(img.shape[0], int(y + size//2))
            crop = img[y1:y2, x1:x2]
            return cv2.resize(crop, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        z_cols[0].image(get_zoom(img_marked, x_links_a_px, y_mid), caption="Zoom Links", use_container_width=True)
        z_cols[1].image(get_zoom(img_marked, x_rechts_a_px, y_mid), caption="Zoom Rechts", use_container_width=True)

        st.subheader("Analyse-Übersicht")
        st.image(img_marked, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 3))
        x_mm = (np.arange(len(kanten_profil)) - x_links_a_px) * mm_per_px
        ax.plot(x_mm, kanten_profil, color='black', lw=1)
        ax.axhline(kanten_sens, color='red', linestyle='--')
        ax.set_ylabel("Kantenstärke")
        st.pyplot(fig)
    else:
        st.error("Keine Kanten gefunden. Bitte die Sensibilität anpassen.")