import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy.signal import find_peaks

# --- HILFSFUNKTIONEN ---

def berechne_mess_toleranz(px_pro_mm, unsicherheit_px=1.5):
    if px_pro_mm > 0:
        return unsicherheit_px / px_pro_mm
    return 0.0

def get_side_analysis(img_rgb, side, kanten_sens, peak_dist):
    h, w = img_rgb.shape[:2]
    center = (w // 2, h // 2)
    
    # Suchbereich festlegen
    if side == "left":
        x_min, x_max = 0, int(w * 0.50)
    else:
        x_min, x_max = int(w * 0.50), w

    # Bildverbesserung
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_enhanced = cv2.createCLAHE(clipLimit=2.5).apply(gray)

    def evaluate_angle(angle):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray_enhanced, M, (w, h), flags=cv2.INTER_LINEAR)
        smooth = cv2.bilateralFilter(rotated, 9, 75, 75)
        grad = np.abs(np.diff(smooth.astype(np.float32), axis=1))
        return np.max(np.mean(grad, axis=0)[x_min:x_max])

    # Winkelsuche
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
    
    # --- ROBUSTE PEAK-SUCHE ---
    # Prominenz auf 0.02 gesenkt, damit auch kleinere Hügel erkannt werden
    peaks, props = find_peaks(
        profil_norm[x_min:x_max], 
        height=kanten_sens, 
        distance=peak_dist, 
        prominence=0.02 
    )
    
    dist_px, img_marked, peaks_rel = 0, opt_img.copy(), []
    
    if len(peaks) >= 2:
        # Die zwei Hügel mit der größten Prominenz (Wichtigkeit) nehmen
        best_idx = np.argsort(props["prominences"])[-2:]
        top_2 = np.sort(peaks[best_idx])
        
        p1_abs, p2_abs = x_min + top_2[0], x_min + top_2[-1]
        dist_px = abs(p2_abs - p1_abs)
        peaks_rel = top_2.tolist()
        
        # Zeichnen: Links (Gelb-Grün), Rechts (Grün-Gelb)
        if side == "left":
            cv2.line(img_marked, (int(p1_abs), 0), (int(p1_abs), h), (255, 255, 0), 6) # Außen
            cv2.line(img_marked, (int(p2_abs), 0), (int(p2_abs), h), (0, 255, 0), 6)   # Innen
        else:
            cv2.line(img_marked, (int(p1_abs), 0), (int(p1_abs), h), (0, 255, 0), 6)   # Innen
            cv2.line(img_marked, (int(p2_abs), 0), (int(p2_abs), h), (255, 255, 0), 6) # Außen
    
    return dist_px, best_angle, img_marked, profil_norm[x_min:x_max], peaks_rel

# --- APP ---
st.set_page_config(page_title="Dual-Precision Pro", layout="wide")
st.sidebar.header("Konfiguration")
ref_mm = st.sidebar.number_input("Referenzbreite Innen (mm)", value=60.00)
sens = st.sidebar.slider("Kanten-Sensitivität", 0.01, 0.50, 0.09)
p_dist = st.sidebar.slider("Mindest-Abstand Kanten (px)", 5, 100, 20)
mm_umdr = st.sidebar.number_input("mm pro Umdrehung", value=0.75)

uploaded_file = st.file_uploader("Bild laden", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img_raw = np.array(Image.open(uploaded_file).convert('RGB'))
    
    # Kalibrierung
    with st.spinner("Analyse läuft..."):
        # Dummy-Kalibrierung zur Ermittlung von px_pro_mm
        gray_init = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)
        sample = np.mean(np.abs(np.diff(gray_init, axis=1)), axis=0)
        p_init, _ = find_peaks(sample/np.max(sample), height=0.4, distance=100)
        px_pro_mm = (p_init[-1] - p_init[0]) / ref_mm if len(p_init) > 1 else 1.0

    col_l, col_r = st.columns(2)
    
    with col_l:
        dist_l_px, ang_l, img_l, prof_l, peaks_l = get_side_analysis(img_raw, "left", sens, p_dist)
        dist_l_mm = dist_l_px / px_pro_mm
        st.metric("Abstand L", f"{dist_l_mm:.3f} mm", f"{ang_l:.1f}°")
        st.image(img_l, use_container_width=True)
        if len(peaks_l) < 2: st.error("Links: Nur 1 Kante gefunden!")
        
        fig_l, ax_l = plt.subplots(figsize=(10, 3))
        ax_l.plot(prof_l, color='cyan')
        ax_l.axhline(sens, color='red', linestyle='--')
        if len(peaks_l) >= 2:
            ax_l.axvline(peaks_l[0], color='yellow'); ax_l.axvline(peaks_l[-1], color='green')
        st.pyplot(fig_l)

    with col_r:
        dist_r_px, ang_r, img_r, prof_r, peaks_r = get_side_analysis(img_raw, "right", sens, p_dist)
        dist_r_mm = dist_r_px / px_pro_mm
        st.metric("Abstand R", f"{dist_r_mm:.3f} mm", f"{ang_r:.1f}°")
        st.image(img_r, use_container_width=True)
        if len(peaks_r) < 2: st.error("Rechts: Nur 1 Kante gefunden!")
        
        fig_r, ax_r = plt.subplots(figsize=(10, 3))
        ax_r.plot(prof_r, color='cyan')
        ax_r.axhline(sens, color='red', linestyle='--')
        if len(peaks_r) >= 2:
            ax_r.axvline(peaks_r[0], color='green'); ax_r.axvline(peaks_r[-1], color='yellow')
        st.pyplot(fig_r)

    # Korrektur
    st.divider()
    diff = dist_l_mm - dist_r_mm
    umdr = round((abs(diff/2)/mm_umdr)*4)/4
    st.header(f"Ergebnis: {umdr} Umdrehungen nach {'RECHTS' if diff/2 > 0 else 'LINKS'}")
