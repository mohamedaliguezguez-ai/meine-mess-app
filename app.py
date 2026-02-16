import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image, ImageOps

# --- HILFSFUNKTIONEN ---

def berechne_mess_toleranz(px_pro_mm, unsicherheit_px=1.5):
    if px_pro_mm > 0:
        return unsicherheit_px / px_pro_mm
    return 0.0

def get_side_metrics(img_rgb, side, search_area_px, px_pro_mm_guess, kanten_sens):
    """
    Findet den optimalen Winkel und den Abstand (Grün-Gelb) für eine spezifische Seite.
    """
    h, w = img_rgb.shape[:2]
    center = (w // 2, h // 2)
    
    # Definieren, wo genau gesucht wird (Linkes Drittel oder Rechtes Drittel)
    if side == "left":
        x_min, x_max = 0, int(w * 0.4)
    else:
        x_min, x_max = int(w * 0.6), w

    def evaluate_angle(angle):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY).astype(np.float32)
        grad = np.abs(np.diff(gray, axis=1))
        profil = np.mean(grad, axis=0)
        # Wir geben die Summe der Peaks in diesem Bereich zurück
        return np.max(profil[x_min:x_max])

    # 1. Iterative Suche nach dem besten Winkel für diese Seite
    angles_coarse = np.arange(-5, 5.5, 0.5)
    best_angle_coarse = angles_coarse[np.argmax([evaluate_angle(a) for a in angles_coarse])]
    
    angles_fine = np.arange(best_angle_coarse - 0.5, best_angle_coarse + 0.6, 0.1)
    best_angle = angles_fine[np.argmax([evaluate_angle(a) for a in angles_fine])]

    # 2. Messung im optimierten Zustand
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    opt_img = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    gray_opt = cv2.cvtColor(opt_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad_opt = np.abs(np.diff(gray_opt, axis=1))
    profil_opt = np.mean(grad_opt, axis=0)
    profil_opt /= (np.max(profil_opt) if np.max(profil_opt) > 0 else 1)

    # Kanten finden im optimierten Profil
    area = profil_opt[x_min:x_max]
    peaks = np.where(area > kanten_sens)[0]
    
    if len(peaks) >= 2:
        # Erster und letzter Peak in diesem Bereich (Grün/Gelb)
        dist_px = peaks[-1] - peaks[0]
        return dist_px, best_angle, profil_opt[x_min:x_max], x_min + peaks[0], x_min + peaks[-1]
    return 0, best_angle, profil_opt[x_min:x_max], 0, 0

# --- APP ---
st.set_page_config(page_title="Dual-Edge Analyse Pro", layout="wide")
st.title("🛠 Dual-Optimierungs Profil-Messung")

# Sidebar
st.sidebar.header("Einstellungen")
ref_weiss_mm = st.sidebar.number_input("Referenzbreite Innen (mm)", value=60.00)
kanten_sens = st.sidebar.slider("Kanten-Sensitivität", 0.01, 0.50, 0.14)
mm_pro_drehung = st.sidebar.number_input("mm pro Umdrehung", value=0.75)

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img_raw = np.array(Image.open(uploaded_file).convert('RGB'))
    h, w = img_raw.shape[:2]

    # Grobe Kalibrierung für px_pro_mm (Zentraler Scan)
    with st.spinner("Initialisiere Kalibrierung..."):
        gray_init = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)
        sample = np.mean(np.abs(np.diff(gray_init[h//2-50:h//2+50, :], axis=1)), axis=0)
        peaks_init = np.where(sample > (np.max(sample)*0.4))[0]
        px_pro_mm = (peaks_init[-1] - peaks_init[0]) / ref_weiss_mm if len(peaks_init) > 1 else 1.0

    # --- DUAL OPTIMIERUNG ---
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.subheader("Linke Seite")
        with st.spinner("Optimiere Links..."):
            dist_l_px, angle_l, prof_l, x_l1, x_l2 = get_side_metrics(img_raw, "left", 0, px_pro_mm, kanten_sens)
            dist_l_mm = dist_l_px / px_pro_mm
            st.metric("Abstand L (Grün-Gelb)", f"{dist_l_mm:.3f} mm", f"Winkel: {angle_l:.1f}°")
            
            fig_l, ax_l = plt.subplots(figsize=(5,2))
            ax_l.plot(prof_l, color='lime')
            ax_l.axhline(kanten_sens, color='red', linestyle='--')
            st.pyplot(fig_l)

    with col_r:
        st.subheader("Rechte Seite")
        with st.spinner("Optimiere Rechts..."):
            dist_r_px, angle_r, prof_r, x_r1, x_r2 = get_side_metrics(img_raw, "right", 0, px_pro_mm, kanten_sens)
            dist_r_mm = dist_r_px / px_pro_mm
            st.metric("Abstand R (Grün-Gelb)", f"{dist_r_mm:.3f} mm", f"Winkel: {angle_r:.1f}°")
            
            fig_r, ax_r = plt.subplots(figsize=(5,2))
            ax_r.plot(prof_r, color='cyan')
            ax_r.axhline(kanten_sens, color='red', linestyle='--')
            st.pyplot(fig_r)

    # --- BERECHNUNG & KORREKTUR ---
    st.divider()
    diff_mm = dist_l_mm - dist_r_mm
    # Die Korrektur ist die halbe Differenz, um das Teil in die Mitte zu rücken
    korrektur_mm = diff_mm / 2.0
    umdrehungen = round((abs(korrektur_mm) / mm_pro_drehung) * 4) / 4
    richtung = "RECHTS" if korrektur_mm > 0 else "LINKS"

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.header("Ergebnis")
        st.metric("Gesamt-Versatz", f"{korrektur_mm:.2f} mm")
        if umdrehungen < 0.25:
            st.success("✅ Bauteil ist korrekt ausgerichtet!")
        else:
            st.warning(f"⚙️ Schraube **{umdrehungen}** Umdr. nach **{richtung}** drehen.")

    with res_col2:
        # Ein kombiniertes Info-Bild (Original ohne Transformation, nur zur Übersicht)
        st.write("Vorschau (Originalbild)")
        st.image(img_raw, use_container_width=True)
