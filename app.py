import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# --- HILFSFUNKTIONEN ---

def berechne_mess_toleranz(px_pro_mm, unsicherheit_px=1.5):
    """Berechnet die physikalische Toleranz der Messung in mm."""
    if px_pro_mm > 0:
        return unsicherheit_px / px_pro_mm
    return 0.0

def get_side_analysis(img_rgb, side, kanten_sens):
    """
    Findet den optimalen Winkel für eine spezifische Seite (links oder rechts)
    und gibt Distanz, Winkel, markiertes Bild, Profil und Peaks zurück.
    """
    h, w = img_rgb.shape[:2]
    center = (w // 2, h // 2)
    
    # Fokus-Bereich definieren (ca. 40% der jeweiligen Seite)
    if side == "left":
        x_min, x_max = 0, int(w * 0.45)
    else:
        x_min, x_max = int(w * 0.55), w

    def evaluate_angle(angle):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY).astype(np.float32)
        grad = np.abs(np.diff(gray, axis=1))
        # Wir maximieren den höchsten Peak im gewählten Bereich
        return np.max(np.mean(grad, axis=0)[x_min:x_max])

    # 1. Iterative Suche (Grob-Suche 0.5° Schritte, Fein-Suche 0.1° Schritte)
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
    
    # Normalisierung für das Diagramm
    max_p = np.max(profil) if np.max(profil) > 0 else 1
    profil_norm = profil / max_p
    
    # Peaks finden, die über der Schwelle liegen
    area_peaks = np.where(profil_norm[x_min:x_max] > kanten_sens)[0]
    
    dist_px = 0
    img_marked = opt_img.copy()
    peaks_rel = []
    
    if len(area_peaks) >= 2:
        p1_idx = area_peaks[0]
        p2_idx = area_peaks[-1]
        peaks_rel = [p1_idx, p2_idx]
        
        p1_abs = x_min + p1_idx
        p2_abs = x_min + p2_idx
        dist_px = abs(p2_abs - p1_abs)
        
        # Linien zeichnen (Gelb=Außen, Grün=Innen)
        if side == "left":
            c_gelb, c_gruen = p1_abs, p2_abs  # Links: Außen ist weiter links
        else:
            c_gruen, c_gelb = p1_abs, p2_abs  # Rechts: Außen ist weiter rechts
            
        cv2.line(img_marked, (int(c_gelb), 0), (int(c_gelb), h), (255, 255, 0), 6) 
        cv2.line(img_marked, (int(c_gruen), 0), (int(c_gruen), h), (0, 255, 0), 6) 
        
    return dist_px, best_angle, img_marked, profil_norm[x_min:x_max], peaks_rel

# --- APP LAYOUT ---
st.set_page_config(page_title="Dual-Optimization Pro", layout="wide")
st.title("🛠 Profil-Mess-App: Duale Seiten-Optimierung")

# --- SEITENLEISTE ---
st.sidebar.header("⚙️ Konfiguration")
ref_weiss_mm = st.sidebar.number_input("Referenzbreite Innen (mm)", value=60.00, step=0.01)
kanten_sens = st.sidebar.slider("Kanten-Sensitivität (Schwelle)", 0.01, 0.50, 0.14)
mm_pro_drehung = st.sidebar.number_input("mm pro Schraubendrehung", value=0.75, step=0.05)

# --- BILD-EINGABE ---
uploaded_file = st.file_uploader("Bild hochladen (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild laden & EXIF Korrektur
    pil_img = ImageOps.exif_transpose(Image.open(uploaded_file))
    if pil_img.width > 2000:
        ratio = 2000 / float(pil_img.width)
        pil_img = pil_img.resize((2000, int(pil_img.height * ratio)), Image.Resampling.LANCZOS)
    
    img_raw = np.array(pil_img.convert('RGB'))
    h, w = img_raw.shape[:2]

    # 1. Globale Skalierung (Initial-Scan in der Mitte)
    with st.spinner("Kalibrierung läuft..."):
        gray_init = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)
        sample = np.mean(np.abs(np.diff(gray_init[h//2-50:h//2+50, :], axis=1)), axis=0)
        sample_norm = sample / (np.max(sample) if np.max(sample) > 0 else 1)
        p_init = np.where(sample_norm > 0.4)[0]
        # Berechnung der px/mm anhand der inneren Kanten im ungedrehten Bild
        px_pro_mm = (p_init[-1] - p_init[0]) / ref_weiss_mm if len(p_init) > 1 else 1.0

    # --- DUAL-SPALTEN ANALYSE ---
    col_l, col_r = st.columns(2)
    
    # LINKER BEREICH
    with col_l:
        st.subheader("⬅️ Linke Seite (Optimiert)")
        with st.spinner("Optimiere Winkel Links..."):
            dist_l_px, ang_l, img_l, prof_l, peaks_l = get_side_analysis(img_raw, "left", kanten_sens)
            dist_l_mm = dist_l_px / px_pro_mm
            
            st.metric("Abstand L (Gelb-Grün)", f"{dist_l_mm:.3f} mm", f"{ang_l:.1f}° Neigung")
            st.image(img_l, caption="Markierte Kanten Links", use_container_width=True)
            
            # Diagramm Links
            fig_l, ax_l = plt.subplots(figsize=(10, 3))
            fig_l.patch.set_facecolor('#0E1117')
            ax_l.set_facecolor('#1e2129')
            ax_l.plot(prof_l, color='cyan', label='Signal')
            ax_l.axhline(kanten_sens, color='red', linestyle='--', label='Schwelle')
            if len(peaks_l) >= 2:
                ax_l.axvline(peaks_l[0], color='yellow', label='Außen')
                ax_l.axvline(peaks_l[-1], color='green', label='Innen')
            ax_l.tick_params(colors='white')
            ax_l.legend()
            st.pyplot(fig_l)

    # RECHTER BEREICH
    with col_r:
        st.subheader("➡️ Rechte Seite (Optimiert)")
        with st.spinner("Optimiere Winkel Rechts..."):
            dist_r_px, ang_r, img_r, prof_r, peaks_r = get_side_analysis(img_raw, "right", kanten_sens)
            dist_r_mm = dist_r_px / px_pro_mm
            
            st.metric("Abstand R (Grün-Gelb)", f"{dist_r_mm:.3f} mm", f"{ang_r:.1f}° Neigung")
            st.image(img_r, caption="Markierte Kanten Rechts", use_container_width=True)
            
            # Diagramm Rechts
            fig_r, ax_r = plt.subplots(figsize=(10, 3))
            fig_r.patch.set_facecolor('#0E1117')
            ax_r.set_facecolor('#1e2129')
            ax_r.plot(prof_r, color='cyan', label='Signal')
            ax_r.axhline(kanten_sens, color='red', linestyle='--', label='Schwelle')
            if len(peaks_r) >= 2:
                ax_r.axvline(peaks_r[0], color='green', label='Innen')
                ax_r.axvline(peaks_r[-1], color='yellow', label='Außen')
            ax_r.tick_params(colors='white')
            ax_r.legend()
            st.pyplot(fig_r)

    # --- FINALERGEBNIS & ANWEISUNG ---
    st.divider()
    diff_mm = dist_l_mm - dist_r_mm
    korrektur_mm = diff_mm / 2.0  # Zentrierungslogik
    umdrehungen = round((abs(korrektur_mm) / mm_pro_drehung) * 4) / 4
    richtung = "RECHTS" if korrektur_mm > 0 else "LINKS"

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.header("📋 Korrektur-Ergebnis")
        st.write(f"Differenz zwischen den Seiten: **{abs(diff_mm):.3f} mm**")
        if umdrehungen < 0.25:
            st.success("✅ **ZENTRIERT:** Keine Korrektur erforderlich.")
        else:
            st.error(f"⚠️ **VERSATZ:** {abs(korrektur_mm):.2f} mm")
            st.subheader(f"⚙️ Schraube {umdrehungen} Umdr. nach {richtung}")
    
    with res_col2:
        st.info(f"""
        **Mess-Details:**
        - Pixel/mm: {px_pro_mm:.2f}
        - Toleranz (±): {berechne_mess_toleranz(px_pro_mm):.3f} mm
        - Optimierungs-Verfahren: Dual-Iterative Peak Search
        """)

else:
    st.info("Bitte laden Sie ein Foto hoch, um die Analyse zu starten.")
