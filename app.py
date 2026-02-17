import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy.signal import find_peaks

# --- HILFSFUNKTIONEN ---

def berechne_mess_toleranz(px_pro_mm, unsicherheit_px=1.5):
    """Berechnet die physikalische Toleranz der Messung in mm."""
    if px_pro_mm > 0:
        return unsicherheit_px / px_pro_mm
    return 0.0

def get_side_analysis(img_rgb, side, kanten_sens, peak_dist):
    """
    Analysiert eine Seite unabhängig:
    1. Kontrast-Boosting (CLAHE) & Rauschfilter (Bilateral).
    2. Iterative Winkelsuche für maximale Peak-Schärfe.
    3. Robuste Peak-Detektion der zwei signifikantesten Kanten.
    """
    h, w = img_rgb.shape[:2]
    center = (w // 2, h // 2)
    
    # Fokus-Bereich (jeweils 50% der Seite für maximale Abdeckung)
    if side == "left":
        x_min, x_max = 0, int(w * 0.50)
    else:
        x_min, x_max = int(w * 0.50), w

    # Vorverarbeitung
    gray_full = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray_full)

    def evaluate_angle(angle):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray_enhanced, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        smooth = cv2.bilateralFilter(rotated, 9, 75, 75)
        grad = np.abs(np.diff(smooth.astype(np.float32), axis=1))
        return np.max(np.mean(grad, axis=0)[x_min:x_max])

    # Iterative Suche
    angles_coarse = np.arange(-5, 5.5, 0.5)
    best_a_c = angles_coarse[np.argmax([evaluate_angle(a) for a in angles_coarse])]
    angles_fine = np.arange(best_a_c - 0.5, best_a_c + 0.6, 0.1)
    best_angle = angles_fine[np.argmax([evaluate_angle(a) for a in angles_fine])]

    # Finales Bild für diese Seite
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    opt_img_rgb = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    opt_gray = cv2.warpAffine(gray_enhanced, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    opt_smooth = cv2.bilateralFilter(opt_gray, 9, 75, 75)
    
    grad_opt = np.abs(np.diff(opt_smooth.astype(np.float32), axis=1))
    profil = np.mean(grad_opt, axis=0)
    profil_norm = profil / (np.max(profil) if np.max(profil) > 0 else 1)
    
    # Peak-Suche mit Mindest-Abstand (verhindert Doppellinien auf einem Hügel)
    peaks, props = find_peaks(
        profil_norm[x_min:x_max], 
        height=kanten_sens, 
        distance=peak_dist, 
        prominence=0.05
    )
    
    dist_px, img_marked, peaks_rel = 0, opt_img_rgb.copy(), []
    
    if len(peaks) >= 2:
        # Die zwei stärksten Peaks nach Prominenz wählen
        best_idx = np.argsort(props["prominences"])[-2:]
        top_2_peaks = np.sort(peaks[best_idx])
        
        p1_abs, p2_abs = x_min + top_2_peaks[0], x_min + top_2_peaks[-1]
        dist_px = abs(p2_abs - p1_abs)
        peaks_rel = top_2_peaks.tolist()
        
        # Farblogik: Außen immer Gelb, Innen immer Grün
        if side == "left":
            c_gelb, c_gruen = p1_abs, p2_abs
        else:
            c_gruen, c_gelb = p1_abs, p2_abs
            
        cv2.line(img_marked, (int(c_gelb), 0), (int(c_gelb), h), (255, 255, 0), 6) 
        cv2.line(img_marked, (int(c_gruen), 0), (int(c_gruen), h), (0, 255, 0), 6) 
        
    return dist_px, best_angle, img_marked, profil_norm[x_min:x_max], peaks_rel

# --- APP LAYOUT ---
st.set_page_config(page_title="Dual-Precision Analyzer", layout="wide")
st.title("🛠 Profil-Mess-App: Dual-Precision Pro")

# --- SEITENLEISTE ---
st.sidebar.header("⚙️ Konfiguration")
ref_mm = st.sidebar.number_input("Referenzbreite Innen (mm)", value=60.00, step=0.01)
kanten_sens = st.sidebar.slider("Kanten-Sensitivität", 0.01, 0.50, 0.09)
peak_dist = st.sidebar.slider("Mindest-Abstand Kanten (px)", 5, 100, 30)
mm_umdr = st.sidebar.number_input("mm pro Umdrehung", value=0.75, step=0.05)

# --- BILD-EINGABE ---
uploaded_file = st.file_uploader("Bild auswählen", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img_raw = np.array(Image.open(uploaded_file).convert('RGB'))
    h, w = img_raw.shape[:2]

    # Kalibrierung
    with st.spinner("Kalibrierung..."):
        gray_init = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)
        clahe_init = cv2.createCLAHE(clipLimit=2.0).apply(gray_init)
        smooth_init = cv2.bilateralFilter(clahe_init, 9, 75, 75)
        sample = np.mean(np.abs(np.diff(smooth_init[h//2-50:h//2+50, :].astype(np.float32), axis=1)), axis=0)
        p_init, _ = find_peaks(sample/(np.max(sample) if np.max(sample)>0 else 1), height=0.4, distance=100)
        px_pro_mm = (p_init[-1] - p_init[0]) / ref_mm if len(p_init) > 1 else 1.0

    # Dual-Analyse
    col_l, col_r = st.columns(2)
    with col_l:
        dist_l_px, ang_l, img_l, prof_l, peaks_l = get_side_analysis(img_raw, "left", kanten_sens, peak_dist)
        dist_l_mm = dist_l_px / px_pro_mm
        st.metric("Abstand Links", f"{dist_l_mm:.3f} mm", f"{ang_l:.1f}°")
        st.image(img_l, use_container_width=True)
        fig_l, ax_l = plt.subplots(figsize=(10, 3))
        ax_l.plot(prof_l, color='cyan')
        ax_l.axhline(kanten_sens, color='red', linestyle='--')
        if len(peaks_l) >= 2:
            ax_l.axvline(peaks_l[0], color='yellow')
            ax_l.axvline(peaks_l[-1], color='green')
        st.pyplot(fig_l)

    with col_r:
        dist_r_px, ang_r, img_r, prof_r, peaks_r = get_side_analysis(img_raw, "right", kanten_sens, peak_dist)
        dist_r_mm = dist_r_px / px_pro_mm
        st.metric("Abstand Rechts", f"{dist_r_mm:.3f} mm", f"{ang_r:.1f}°")
        st.image(img_r, use_container_width=True)
        fig_r, ax_r = plt.subplots(figsize=(10, 3))
        ax_r.plot(prof_r, color='cyan')
        ax_r.axhline(kanten_sens, color='red', linestyle='--')
        if len(peaks_r) >= 2:
            ax_r.axvline(peaks_r[0], color='green')
            ax_r.axvline(peaks_r[-1], color='yellow')
        st.pyplot(fig_r)

    # Sidebar Info (Hier wurde der NameError gefixt)
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Winkel L: {ang_l:.2f}° | R: {ang_r:.2f}°")

    # Ergebnis
    st.divider()
    diff_mm = dist_l_mm - dist_r_mm
    korrektur_mm = diff_mm / 2.0
    umdrehungen = round((abs(korrektur_mm) / mm_umdr) * 4) / 4
    res_c1, res_c2 = st.columns(2)
    with res_c1:
        st.header("📋 Ergebnis")
        if umdrehungen < 0.25: st.success("✅ Zentriert!")
        else: st.error(f"Schraube {umdrehungen} Umdr. nach {'RECHTS' if korrektur_mm > 0 else 'LINKS'}")
    with res_c2:
        st.info(f"Differenz: {abs(diff_mm):.3f} mm\n\nPräzision: {berechne_mess_toleranz(px_pro_mm):.3f} mm")
