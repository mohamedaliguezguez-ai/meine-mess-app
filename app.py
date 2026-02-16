import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# --- HILFSFUNKTIONEN ---

def berechne_mess_toleranz(px_pro_mm, unsicherheit_px=1.5):
    if px_pro_mm > 0:
        return unsicherheit_px / px_pro_mm
    return 0.0

def get_side_analysis(img_rgb, side, kanten_sens):
    """
    Findet den optimalen Winkel für eine Seite und gibt das 
    optimierte Bild mit eingezeichneten Linien zurück.
    """
    h, w = img_rgb.shape[:2]
    center = (w // 2, h // 2)
    
    # Fokus-Bereich definieren
    if side == "left":
        x_min, x_max = 0, int(w * 0.45)
    else:
        x_min, x_max = int(w * 0.55), w

    def evaluate_angle(angle):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY).astype(np.float32)
        grad = np.abs(np.diff(gray, axis=1))
        # Peak-Stärke in der Fokus-Zone
        return np.max(np.mean(grad, axis=0)[x_min:x_max])

    # 1. Iterative Suche (Grob 0.5°, Fein 0.1°)
    angles_coarse = np.arange(-5, 5.5, 0.5)
    best_a_c = angles_coarse[np.argmax([evaluate_angle(a) for a in angles_coarse])]
    angles_fine = np.arange(best_a_c - 0.5, best_a_c + 0.6, 0.1)
    best_angle = angles_fine[np.argmax([evaluate_angle(a) for a in angles_fine])]

    # 2. Finales Bild für diese Seite erstellen
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    opt_img = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Kanten im Profil finden
    gray_opt = cv2.cvtColor(opt_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad_opt = np.abs(np.diff(gray_opt, axis=1))
    profil = np.mean(grad_opt, axis=0)
    profil /= (np.max(profil) if np.max(profil) > 0 else 1)
    
    area_peaks = np.where(profil[x_min:x_max] > kanten_sens)[0]
    
    dist_px = 0
    img_marked = opt_img.copy()
    
    if len(area_peaks) >= 2:
        p1 = x_min + area_peaks[0]
        p2 = x_min + area_peaks[-1]
        dist_px = abs(p2 - p1)
        
        # Linien zeichnen (Gelb=Außen, Grün=Innen)
        if side == "left":
            c_gelb, c_gruen = p1, p2  # Links ist Außen weiter links (kleineres X)
        else:
            c_gruen, c_gelb = p1, p2  # Rechts ist Außen weiter rechts (größeres X)
            
        cv2.line(img_marked, (int(c_gelb), 0), (int(c_gelb), h), (255, 255, 0), 4) # Gelb
        cv2.line(img_marked, (int(c_gruen), 0), (int(c_gruen), h), (0, 255, 0), 4) # Grün
        
    return dist_px, best_angle, img_marked, profil[x_min:x_max]

# --- APP ---
st.set_page_config(page_title="Dual-Precision Mess-App", layout="wide")
st.title("🛠 Dual-Optimierung: Getrennte Seiten-Analyse")

# Sidebar
st.sidebar.header("Konfiguration")
ref_mm = st.sidebar.number_input("Referenzbreite Innen (mm)", value=60.00)
sens = st.sidebar.slider("Kanten-Sensitivität", 0.01, 0.50, 0.14)
mm_umdr = st.sidebar.number_input("mm pro Umdrehung", value=0.75)

uploaded_file = st.file_uploader("Bild auswählen", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img_raw = np.array(Image.open(uploaded_file).convert('RGB'))
    h, w = img_raw.shape[:2]

    # 1. Globale Kalibrierung (Zentrum)
    gray_init = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)
    sample = np.mean(np.abs(np.diff(gray_init[h//2-50:h//2+50, :], axis=1)), axis=0)
    p_init = np.where(sample > (np.max(sample)*0.4))[0]
    px_pro_mm = (p_init[-1] - p_init[0]) / ref_mm if len(p_init) > 1 else 1.0

    # --- ZWEI-SPALTEN LAYOUT ---
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.subheader("📍 Optimierung Links")
        with st.spinner("Berechne links..."):
            dist_l_px, ang_l, img_l, prof_l = get_side_analysis(img_raw, "left", sens)
            dist_l_mm = dist_l_px / px_pro_mm
            st.metric("Abstand Links", f"{dist_l_mm:.3f} mm", f"Winkel: {ang_l:.1f}°")
            st.image(img_l, caption="Optimierte linke Kanten (Gelb=Außen, Grün=Innen)", use_container_width=True)

    with col_r:
        st.subheader("📍 Optimierung Rechts")
        with st.spinner("Berechne rechts..."):
            dist_r_px, ang_r, img_r, prof_r = get_side_analysis(img_raw, "right", sens)
            dist_r_mm = dist_r_px / px_pro_mm
            st.metric("Abstand Rechts", f"{dist_r_mm:.3f} mm", f"Winkel: {ang_r:.1f}°")
            st.image(img_r, caption="Optimierte rechte Kanten (Gelb=Außen, Grün=Innen)", use_container_width=True)

    # --- KORREKTUR-BERECHNUNG ---
    st.divider()
    diff_mm = dist_l_mm - dist_r_mm
    korrektur_mm = diff_mm / 2.0
    umdrehungen = round((abs(korrektur_mm) / mm_umdr) * 4) / 4
    richtung = "RECHTS" if korrektur_mm > 0 else "LINKS"

    res_l, res_r = st.columns(2)
    with res_l:
        st.header("⚙️ Korrektur-Anweisung")
        if umdrehungen < 0.25:
            st.success("✅ Das Bauteil ist perfekt zentriert!")
        else:
            st.warning(f"Drehe die Schraube **{umdrehungen}** Umdr. nach **{richtung}**.")
    
    with res_r:
        st.info(f"**Details:**\n- Differenz L/R: {abs(diff_mm):.3f} mm\n- Berechneter Versatz: {korrektur_mm:.3f} mm")
