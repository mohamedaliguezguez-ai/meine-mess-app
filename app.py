import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image, ImageOps

# --- HILFSFUNKTIONEN ---

def berechne_mess_toleranz(px_pro_mm, unsicherheit_px=1.5):
    """Berechnet die physikalische Toleranz der Messung in mm."""
    if px_pro_mm > 0:
        return unsicherheit_px / px_pro_mm
    return 0.0

def iterative_straighten(img_rgb, ref_mm):
    """
    Sucht iterativ den optimalen Winkel durch Maximierung des Gradienten-Peaks.
    Bereich: +/- 5° | Schritte: 0.5° danach 0.1°
    """
    h, w = img_rgb.shape[:2]
    center = (w // 2, h // 2)
    
    # 1. Grobe Skalierung schätzen, um den Auswertungsbereich in Pixeln zu definieren
    # Wir nehmen an, dass das Bauteil grob in der Mitte liegt.
    gray_init = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sample_row = gray_init[h // 2, :]
    grad_init = np.abs(np.diff(sample_row.astype(np.float32)))
    peaks = np.where(grad_init > (np.max(grad_init) * 0.5))[0]
    
    if len(peaks) >= 2:
        px_pro_mm_rough = (peaks[-1] - peaks[0]) / ref_mm
    else:
        px_pro_mm_rough = w / (ref_mm * 1.5) # Notfall-Schätzung

    # Auswertungsbereich: Mitte +/- (Ref/2 + 5mm)
    half_width_px = int((ref_mm / 2 + 5) * px_pro_mm_rough)
    x_start = max(0, center[0] - half_width_px)
    x_end = min(w, center[0] + half_width_px)

    def get_max_gradient(angle):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # Gradienten-Profil berechnen
        gray_rot = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY).astype(np.float32)
        grad = np.abs(np.diff(gray_rot, axis=1))
        profil = np.mean(grad, axis=0)
        # Suche Peak nur im spezifizierten Auswertungsbereich
        return np.max(profil[x_start:x_end]) if x_end > x_start else 0

    # --- SCHRITT 1: Grob-Suche (-5° bis +5°, 0.5° Schritte) ---
    angles_coarse = np.arange(-5, 5.5, 0.5)
    peaks_coarse = [get_max_gradient(a) for a in angles_coarse]
    best_idx = np.argmax(peaks_coarse)
    best_angle_coarse = angles_coarse[best_idx]

    # --- SCHRITT 2: Fein-Suche (+/- 0.5° um Bestwert, 0.1° Schritte) ---
    angles_fine = np.arange(best_angle_coarse - 0.5, best_angle_coarse + 0.6, 0.1)
    peaks_fine = [get_max_gradient(a) for a in angles_fine]
    best_angle_final = angles_fine[np.argmax(peaks_fine)]

    # Finales Bild rotieren
    M_final = cv2.getRotationMatrix2D(center, best_angle_final, 1.0)
    img_final = cv2.warpAffine(img_rgb, M_final, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return img_final, best_angle_final

# --- APP KONFIGURATION ---
st.set_page_config(page_title="Präzisions-Analyse Pro", layout="centered")
st.title("🛠 Profil-Mess-App Pro")

# --- SEITENLEISTE ---
st.sidebar.header("Configuration")
orientierung = st.sidebar.radio("Component position:", ("Horizontal (Lying)", "Vertical (Standing)"))
kanten_sens = st.sidebar.slider("Edge Sensitivity", 0.01, 0.50, 0.14, 0.01)
ref_weiss_mm = st.sidebar.number_input("Reference Width Inside (mm)", value=60.00)
such_offset_mm = st.sidebar.slider("Search Offset (mm)", 0.5, 20.0, 5.0, 0.5)
mm_pro_drehung = st.sidebar.number_input("mm per Revolution", value=0.75)

st.sidebar.markdown("---")
do_auto_level = st.sidebar.checkbox("Iterative Begradigung aktivieren", value=True)
manual_angle = st.sidebar.slider("Fein-Justierung (°)", -5.0, 5.0, 0.0, 0.05)

# --- BILD-EINGABE ---
input_method = st.radio("Image source:", ("Use camera", "Screenshot / Upload file"))
uploaded_file = st.camera_input("Foto") if input_method == "Use camera" else st.file_uploader("Select image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Vorbereitung & Laden
    pil_img = ImageOps.exif_transpose(Image.open(uploaded_file))
    if pil_img.width > 2000:
        ratio = 2000 / float(pil_img.width)
        pil_img = pil_img.resize((2000, int(pil_img.height * ratio)), Image.Resampling.LANCZOS)
    
    img_rgb = np.array(pil_img.convert('RGB'))
    
    # Grund-Ausrichtung
    if orientierung == "Horizontal (Lying)":
        img_work = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)
    else:
        img_work = img_rgb.copy()

    # 2. Iterative Begradigung
    auto_angle = 0.0
    if do_auto_level:
        with st.spinner("Optimiere Bildwinkel (iterativ)..."):
            img_rot, auto_angle = iterative_straighten(img_work, ref_weiss_mm)
    else:
        img_rot = img_work

    # Manuelle Korrektur obendrauf
    if manual_angle != 0.0:
        h_r, w_r = img_rot.shape[:2]
        M_man = cv2.getRotationMatrix2D((w_r // 2, h_r // 2), manual_angle, 1.0)
        img_rot = cv2.warpAffine(img_rot, M_man, (w_r, h_r), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    st.sidebar.info(f"Korrektur-Winkel: {auto_angle + manual_angle:.2f}°")

    # 3. Analyse
    gray = (0.2989 * img_rot[:,:,0] + 0.5870 * img_rot[:,:,1] + 0.1140 * img_rot[:,:,2]).astype(np.float64)
    # Glättung für stabileres Profil
    gray_smooth = convolve2d(gray, np.ones((3, 3))/9.0, mode='same')
    h_grad = np.abs(np.diff(gray_smooth, axis=1))
    kanten_profil = np.mean(h_grad, axis=0)
    
    if np.max(kanten_profil) > 0:
        kanten_profil = kanten_profil / np.max(kanten_profil)

    # 4. Kanten finden
    img_center = kanten_profil.shape[0] // 2
    
    # Innenkanten (Grün)
    suche_r = np.where(kanten_profil[img_center:] > kanten_sens)[0]
    x_rechts_w_px = (img_center + suche_r[0]) if len(suche_r) > 0 else img_center
    suche_l = np.where(kanten_profil[:img_center][::-1] > kanten_sens)[0]
    x_links_w_px = (img_center - suche_l[0]) if len(suche_l) > 0 else img_center

    if x_rechts_w_px > x_links_w_px:
        px_pro_mm = (x_rechts_w_px - x_links_w_px) / ref_weiss_mm
        such_offset_px = int(such_offset_mm * px_pro_mm)
        
        # Außenkanten (Gelb)
        start_r_a = min(len(kanten_profil)-1, x_rechts_w_px + such_offset_px)
        idx_r_a = np.where(kanten_profil[x_rechts_w_px+5:start_r_a+1][::-1] > kanten_sens)[0]
        x_rechts_a_px = (start_r_a - idx_r_a[0]) if len(idx_r_a) > 0 else start_r_a

        start_l_a = max(0, x_links_w_px - such_offset_px)
        idx_l_a = np.where(kanten_profil[start_l_a:x_links_w_px-5] > kanten_sens)[0]
        x_links_a_px = (start_l_a + idx_l_a[0]) if len(idx_l_a) > 0 else start_l_a

        # 5. Berechnung
        zentrum_ist_px = (x_links_w_px + x_rechts_w_px) / 2.0
        zentrum_soll_px = (x_links_a_px + x_rechts_a_px) / 2.0
        abweichung_mm = (zentrum_ist_px - zentrum_soll_px) / px_pro_mm
        
        raw_umdr = abs(abweichung_mm) / mm_pro_drehung
        umdrehungen = round(raw_umdr * 4) / 4
        anweisung = "RIGHT" if abweichung_mm <= 0 else "LEFT"

        # --- ERGEBNIS ANZEIGE ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Deviation", f"{abs(abweichung_mm):.2f} mm")
        col2.metric("Tolerance (±)", f"{berechne_mess_toleranz(px_pro_mm):.2f} mm")
        col3.metric("Correction", f"{umdrehungen} Umdr.")

        if umdrehungen < 0.25:
            st.info("✅ Zentriert (Abweichung < 0.2 mm)")
        else:
            st.success(f"⚙️ Schraube **{umdrehungen}** Umdr. nach **{anweisung}** drehen.")

        # --- VISUALISIERUNG ---
        img_marked = img_rot.copy()
        h_img = img_marked.shape[0]
        for x, c, w in [(x_links_a_px, (255,255,0), 2), (x_rechts_a_px, (255,255,0), 2), 
                        (x_links_w_px, (0,255,0), 2), (x_rechts_w_px, (0,255,0), 2),
                        (zentrum_soll_px, (0,0,255), 2), (zentrum_ist_px, (255,0,0), 2)]:
            cv2.line(img_marked, (int(x), 0), (int(x), h_img), c, w)

        st.subheader("Analysis Overview")
        st.image(img_marked, use_container_width=True)

        # --- DIAGRAMM ---
        st.divider()
        st.subheader("📊 Kanten-Signal-Analyse")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.patch.set_facecolor('#0E1117') 
        ax.set_facecolor('#1e2129')
        ax.plot(kanten_profil, color='#00d1ff', label='Kontrast-Stärke')
        ax.axhline(y=kanten_sens, color='red', linestyle='--', label='Schwelle')
        
        # Kantenmarkierungen im Plot
        for x, col, lab in [(x_links_w_px, 'lime', 'Innen'), (x_links_a_px, 'orange', 'Außen')]:
            ax.axvline(x=x, color=col, alpha=0.5, label=lab)
        ax.axvline(x=x_rechts_w_px, color='lime', alpha=0.5)
        ax.axvline(x=x_rechts_a_px, color='orange', alpha=0.5)

        ax.set_ylim(0, 1.1)
        ax.tick_params(colors='white')
        ax.legend(loc='upper right', facecolor='#0E1117', labelcolor='white')
        st.pyplot(fig)
    else:
        st.error("Keine Kanten gefunden. Bitte Sensitivität anpassen.")
