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
            cv2.line(img_marked, (int(p1_abs), 0), (int(p1_abs), h), (255, 255, 0), 2) # Außen
            cv2.line(img_marked, (int(p2_abs), 0), (int(p2_abs), h), (0, 255, 0), 2)   # Innen
        else:
            cv2.line(img_marked, (int(p1_abs), 0), (int(p1_abs), h), (0, 255, 0), 2)   # Innen
            cv2.line(img_marked, (int(p2_abs), 0), (int(p2_abs), h), (255, 255, 0), 2) # Außen
    
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
# --- KALIBRIERUNG (PX ZU MM) MIT VISUELLER KONTROLLE ---
    with st.spinner("Kalibrierung läuft..."):
        gray_init = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)
        # Verbesserung für die Kalibrierung
        clahe_init = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray_init)
        smooth_init = cv2.bilateralFilter(clahe_init, 9, 75, 75)
        
        sample = np.mean(np.abs(np.diff(smooth_init.astype(np.float32), axis=1)), axis=0)
        sample_norm = sample / (np.max(sample) if np.max(sample) > 0 else 1)
        
        # Suche die inneren Referenz-Kanten
        p_init, _ = find_peaks(sample_norm, height=0.4, distance=100)
        
        if len(p_init) >= 2:
            # Berechnung des Maßstabs
            pixel_abstand = p_init[-1] - p_init[0]
            px_pro_mm = pixel_abstand / ref_mm
            
            # --- VISUALISIERUNG DER KALIBRIERUNG ---
            st.subheader("📏 Kalibrierungs-Check (Referenz-Abstand)")
            img_calib = img_raw.copy()
            h_c, w_c = img_calib.shape[:2]
            
            # Zeichne zwei dicke blaue Linien auf die erkannten Referenz-Punkte
            cv2.line(img_calib, (int(p_init[0]), 0), (int(p_init[0]), h_c), (255, 0, 0), 12) 
            cv2.line(img_calib, (int(p_init[-1]), 0), (int(p_init[-1]), h_c), (255, 0, 0), 12) 
            
            # Bild anzeigen
            st.image(img_calib, 
                     caption=f"Kalibrierung: {pixel_abstand} Pixel entsprechen {ref_mm} mm (Blau markiert)", 
                     use_container_width=True)
            
            st.info(f"Maßstab: **{px_pro_mm:.2f} px/mm**")
        else:
            px_pro_mm = 1.0
            st.error("❌ Kalibrierung fehlgeschlagen: Konnte keine zwei Referenz-Kanten finden!")

    # Korrektur
    st.divider()
    diff = dist_l_mm - dist_r_mm
    umdr = round((abs(diff/2)/mm_umdr)*4)/4
    st.header(f"Ergebnis: {umdr} Umdrehungen nach {'RECHTS' if diff/2 > 0 else 'LINKS'}")


