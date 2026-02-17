import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy.signal import find_peaks

# --- 1. HILFSFUNKTIONEN ---

def berechne_mess_toleranz(px_pro_mm, unsicherheit_px=1.5):
    if px_pro_mm > 0:
        return unsicherheit_px / px_pro_mm
    return 0.0

def get_side_analysis(img_rgb, side, kanten_sens, peak_dist):
    h, w = img_rgb.shape[:2]
    center = (w // 2, h // 2)
    
    # Suchbereich festlegen (50/50 Split)
    if side == "left":
        x_min, x_max = 0, int(w * 0.50)
    else:
        x_min, x_max = int(w * 0.50), w

    # Bildverbesserung (CLAHE + Bilateral)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_enhanced = cv2.createCLAHE(clipLimit=2.5).apply(gray)

    def evaluate_angle(angle):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray_enhanced, M, (w, h), flags=cv2.INTER_LINEAR)
        smooth = cv2.bilateralFilter(rotated, 9, 75, 75)
        grad = np.abs(np.diff(smooth.astype(np.float32), axis=1))
        return np.max(np.mean(grad, axis=0)[x_min:x_max])

    # Winkelsuche (Grob & Fein)
    angles = np.arange(-5, 5.5, 0.5)
    best_a = angles[np.argmax([evaluate_angle(a) for a in angles])]
    fine_angles = np.arange(best_a - 0.5, best_a + 0.6, 0.1)
    best_angle = fine_angles[np.argmax([evaluate_angle(a) for a in fine_angles])]

    # Analyse im Bestwinkel
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    opt_img = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_CUBIC)
    opt_gray = cv2.warpAffine(gray_enhanced, M, (w, h), flags=cv2.INTER_CUBIC)
    opt_smooth = cv2.bilateralFilter(opt_gray, 9, 75, 75)
    
    grad = np.abs(np.diff(opt_smooth.astype(np.float32), axis=1))
    profil = np.mean(grad, axis=0)
    profil_norm = profil / (np.max(profil) if np.max(profil) > 0 else 1)
    
    # Peak-Suche
    peaks, props = find_peaks(profil_norm[x_min:x_max], height=kanten_sens, distance=peak_dist, prominence=0.02)
    
    dist_px, img_marked, peaks_rel = 0, opt_img.copy(), []
    
    if len(peaks) >= 2:
        best_idx = np.argsort(props["prominences"])[-2:]
        top_2 = np.sort(peaks[best_idx])
        p1_abs, p2_abs = x_min + top_2[0], x_min + top_2[-1]
        dist_px = abs(p2_abs - p1_abs)
        peaks_rel = top_2.tolist()
        
        if side == "left":
            cv2.line(img_marked, (int(p1_abs), 0), (int(p1_abs), h), (255, 255, 0), 1) # Außen
            cv2.line(img_marked, (int(p2_abs), 0), (int(p2_abs), h), (0, 255, 0), 1)   # Innen
        else:
            cv2.line(img_marked, (int(p1_abs), 0), (int(p1_abs), h), (0, 255, 0), 1)   # Innen
            cv2.line(img_marked, (int(p2_abs), 0), (int(p2_abs), h), (255, 255, 0), 1) # Außen
    
    return dist_px, best_angle, img_marked, profil_norm[x_min:x_max], peaks_rel

# --- 2. APP KONFIGURATION ---
st.set_page_config(page_title="Dual-Precision Pro", layout="wide")
st.title("🛠 Profil-Mess-App Pro")

# --- 3. SEITENLEISTE ---
st.sidebar.header("Konfiguration")
ref_mm = st.sidebar.number_input("Referenzbreite Innen (mm)", value=60.00)
sens = st.sidebar.slider("Kanten-Sensitivität", 0.01, 0.50, 0.09)
p_dist = st.sidebar.slider("Mindest-Abstand Kanten (px)", 5, 100, 20)
mm_umdr = st.sidebar.number_input("mm pro Umdrehung", value=0.75)

# --- 4. BILD-UPLOAD ---
uploaded_file = st.file_uploader("Bild laden", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Bild laden & EXIF-Korrektur (Handyfotos drehen)
    pil_img = ImageOps.exif_transpose(Image.open(uploaded_file))
    img_raw = np.array(pil_img.convert('RGB'))
    h, w = img_raw.shape[:2]

    # --- 5. VISUELLE KALIBRIERUNG ---
    st.divider()
    with st.spinner("Kalibrierung läuft..."):
        gray_init = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)
        clahe_init = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray_init)
        smooth_init = cv2.bilateralFilter(clahe_init, 9, 75, 75)
        
        sample = np.mean(np.abs(np.diff(smooth_init.astype(np.float32), axis=1)), axis=0)
        sample_norm = sample / (np.max(sample) if np.max(sample) > 0 else 1)
        
        # Referenz-Kanten suchen
        p_init, _ = find_peaks(sample_norm, height=0.4, distance=100)
        
        if len(p_init) >= 2:
            pixel_abstand = p_init[-1] - p_init[0]
            px_pro_mm = pixel_abstand / ref_mm
            
            # Anzeige der Kalibrierung
            st.subheader("📏 Kalibrierungs-Check (Innere Kanten)")
            img_calib = img_raw.copy()
            cv2.line(img_calib, (int(p_init[0]), 0), (int(p_init[0]), h), (255, 0, 0), 2) 
            cv2.line(img_calib, (int(p_init[-1]), 0), (int(p_init[-1]), h), (255, 0, 0), 2) 
            st.image(img_calib, caption=f"Referenz-Check: {pixel_abstand}px = {ref_mm}mm (Blau markiert)", use_container_width=True)
            st.info(f"Maßstab ermittelt: **{px_pro_mm:.2f} px/mm**")
        else:
            px_pro_mm = 1.0
            st.error("❌ Kalibrierung fehlgeschlagen! Bitte Kanten besser ausleuchten.")

    # --- 6. DUAL-ANALYSE ---
    st.divider()
    col_l, col_r = st.columns(2)
    
    with col_l:
        dist_l_px, ang_l, img_l, prof_l, peaks_l = get_side_analysis(img_raw, "left", sens, p_dist)
        dist_l_mm = dist_l_px / px_pro_mm
        st.metric("Abstand Links", f"{dist_l_mm:.3f} mm", f"{ang_l:.1f}°")
        st.image(img_l, use_container_width=True)
        
        fig_l, ax_l = plt.subplots(figsize=(10, 3))
        ax_l.plot(prof_l, color='cyan')
        ax_l.axhline(sens, color='red', linestyle='--')
        if len(peaks_l) >= 2:
            ax_l.axvline(peaks_l[0], color='yellow'); ax_l.axvline(peaks_l[-1], color='green')
        st.pyplot(fig_l)

    with col_r:
        dist_r_px, ang_r, img_r, prof_r, peaks_r = get_side_analysis(img_raw, "right", sens, p_dist)
        dist_r_mm = dist_r_px / px_pro_mm
        st.metric("Abstand Rechts", f"{dist_r_mm:.3f} mm", f"{ang_r:.1f}°")
        st.image(img_r, use_container_width=True)
        
        fig_r, ax_r = plt.subplots(figsize=(10, 3))
        ax_r.plot(prof_r, color='cyan')
        ax_r.axhline(sens, color='red', linestyle='--')
        if len(peaks_r) >= 2:
            ax_r.axvline(peaks_r[0], color='green'); ax_r.axvline(peaks_r[-1], color='yellow')
        st.pyplot(fig_r)

    # --- 7. ERGEBNIS & ANWEISUNG ---
    st.divider()
    diff = dist_l_mm - dist_r_mm
    korrektur_mm = diff / 2.0
    umdr = round((abs(korrektur_mm) / mm_umdr) * 4) / 4
    
    res_c1, res_c2 = st.columns(2)
    with res_c1:
        st.header("📋 Korrektur-Ergebnis")
        if umdr < 0.25:
            st.success("✅ Das Bauteil ist korrekt zentriert!")
        else:
            richtung = "RECHTS" if korrektur_mm > 0 else "LINKS"
            st.error(f"Schraube **{umdr} Umdr.** nach **{richtung}** drehen.")
    
    with res_c2:
        st.info(f"Differenz: {abs(diff):.3f} mm\n\nPräzision: {berechne_mess_toleranz(px_pro_mm):.3f} mm")

