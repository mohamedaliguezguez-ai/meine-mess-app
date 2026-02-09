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

# NEU: Auswahl der Ausrichtung
orientierung = st.sidebar.radio(
    "Wie liegt das Bauteil im Bild?",
    ("Horizontal (Liegend)", "Vertikal (Stehend)"),
    help="Wähle 'Horizontal', wenn das Bauteil quer im Bild liegt (z.B. bei Screenshots)."
)

kanten_sens = st.sidebar.slider(
    "Kanten-Sensibilität", 
    min_value=0.01, max_value=0.30, value=0.14, step=0.01
)
ref_weiss_mm = st.sidebar.number_input("Referenzbreite Weiß (mm)", value=60.00)
such_offset_mm = st.sidebar.slider("Such-Offset (mm)", 3, 30, 10)
mm_pro_drehung = st.sidebar.number_input("mm pro Umdrehung", value=0.75)

# --- BILD-EINGABE ---
input_method = st.radio("Bildquelle:", ("Kamera nutzen", "Screenshot / Datei hochladen"))

uploaded_file = None
if input_method == "Kamera nutzen":
    uploaded_file = st.camera_input("Foto aufnehmen")
else:
    uploaded_file = st.file_uploader("Screenshot oder Bilddatei auswählen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Bild laden & Handy-Drehung korrigieren
    pil_img = Image.open(uploaded_file)
    pil_img = ImageOps.exif_transpose(pil_img)
    
    # 2. Größenoptimierung (S25 Ultra / High-Res Screenshots)
    if pil_img.width > 2000:
        ratio = 2000 / float(pil_img.width)
        new_h = int(float(pil_img.height) * float(ratio))
        pil_img = pil_img.resize((2000, new_h), Image.Resampling.LANCZOS)
    
    img_rgb = np.array(pil_img.convert('RGB'))
    
    # --- LOGIK FÜR DIE DREHUNG ---
    # Unser Scan braucht das Profil immer vertikal (stehend), 
    # damit er von links nach rechts messen kann.
    if orientierung == "Horizontal (Liegend)":
        # Wenn es liegt, drehen wir es einmal um 90 Grad
        img_rot = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)
    else:
        # Wenn es bereits steht, lassen wir es so
        img_rot = img_rgb

    # --- MESS-ANALYSE ---
    gray = (0.2989 * img_rot[:,:,0] + 0.5870 * img_rot[:,:,1] + 0.1140 * img_rot[:,:,2]).astype(np.float64)
    kernel = np.ones((5, 5), np.float64) / 25.0
    gray_smooth = convolve2d(gray, kernel, mode='same')
    h_grad = np.abs(np.diff(gray_smooth, axis=1))
    h_grad = np.hstack((h_grad, np.zeros((h_grad.shape[0], 1))))
    kanten_profil = np.mean(h_grad, axis=0)
    kanten_profil = kanten_profil / np.max(kanten_profil)

    # Innenkanten (Grün)
    bw_weiss = gray > 180
    x_proj_w = np.sum(bw_weiss, axis=0)
    weiss_idx = np.where(x_proj_w > (np.max(x_proj_w) * 0.5))[0]
    
    if len(weiss_idx) > 0:
        x_links_w_px = weiss_idx[0]; x_rechts_w_px = weiss_idx[-1]
        px_pro_mm = (x_rechts_w_px - x_links_w_px) / ref_weiss_mm
        offset_px = int(round(such_offset_mm * px_pro_mm))
        mm_per_px = 1.0 / px_pro_mm

        # Außenkanten (Gelb)
        start_l = max(0, x_links_w_px - offset_px)
        suche_l_bereich = kanten_profil[start_l : x_links_w_px - 5]
        idx_l_list = np.where(suche_l_bereich > kanten_sens)[0]
        x_links_a_px = start_l + idx_l_list[0] if len(idx_l_list) > 0 else 0

        start_r = min(len(kanten_profil) - 1, x_rechts_w_px + offset_px)
        suche_r_bereich = kanten_profil[x_rechts_w_px + 5 : start_r + 1]
        idx_r_list = np.where(suche_r_bereich[::-1] > kanten_sens)[0]
        x_rechts_a_px = start_r - idx_r_list[0] if len(idx_r_list) > 0 else len(kanten_profil)-1

        # Berechnung der Korrektur
        zentrum_ist_px = (x_links_w_px + x_rechts_w_px) / 2.0
        zentrum_soll_px = (x_links_a_px + x_rechts_a_px) / 2.0
        abweichung_mm = (zentrum_ist_px - zentrum_soll_px) * mm_per_px
        umdrehungen = abs(abweichung_mm) / mm_pro_drehung
        anweisung = "RECHTS herum" if abweichung_mm <= 0 else "LINKS herum"

        # Ergebnisse anzeigen
        col1, col2 = st.columns(2)
        col1.metric("Abweichung", f"{abs(abweichung_mm):.2f} mm")
        col2.metric("Korrektur", f"{umdrehungen:.2f} Umdr.")
        st.success(f"Drehe die Schraube **{anweisung}**.")

        # Zeichnen
        img_marked = img_rot.copy()
        h_img, w_img, _ = img_marked.shape
        cv2.line(img_marked, (int(x_links_a_px), 0), (int(x_links_a_px), h_img), (255, 255, 0), 6)
        cv2.line(img_marked, (int(x_rechts_a_px), 0), (int(x_rechts_a_px), h_img), (255, 255, 0), 6)
        cv2.line(img_marked, (int(x_links_w_px), 0), (int(x_links_w_px), h_img), (0, 255, 0), 6)
        cv2.line(img_marked, (int(x_rechts_w_px), 0), (int(x_rechts_w_px), h_img), (0, 255, 0), 6)
        
        st.image(img_marked, caption="Analyse (wird für Messung intern immer aufrecht gestellt)", use_container_width=True)
    else:
        st.error("Weißer Bereich ($90\text{ mm}$) nicht gefunden.")