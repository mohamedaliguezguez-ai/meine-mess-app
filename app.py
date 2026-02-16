import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image, ImageOps

def berechne_mess_toleranz(px_pro_mm, unsicherheit_px=1.5):
    """Berechnet die physikalische Toleranz der Messung in mm."""
    if px_pro_mm > 0:
        return unsicherheit_px / px_pro_mm
    return 0.0
    
def auto_begradigen(img_rgb):
    """Richtet das Bild aus und ignoriert dabei den äußeren Rand."""
    h, w = img_rgb.shape[:2]
    
    # 1. Wir erstellen eine Kopie nur für die Analyse und schneiden 5% vom Rand weg
    # Damit der Bildrahmen nicht als Kontur erkannt wird.
    margin_h = int(h * 0.05)
    margin_w = int(w * 0.05)
    analyse_zone = img_rgb[margin_h:h-margin_h, margin_w:w-margin_w]
    
    gray = cv2.cvtColor(analyse_zone, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Kanten finden (Canny ist hier meist besser als Threshold)
    edged = cv2.Canny(blur, 50, 150)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img_rgb, 0.0
    
    # Die größte Kontur im inneren Bereich suchen
    c = max(contours, key=cv2.contourArea)
    
    # Winkel des umschließenden Rechtecks berechnen
    rect = cv2.minAreaRect(c)
    angle = rect[-1]
    
    # OpenCV Winkel-Logik korrigieren
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle
        
    # 2. Die Rotation auf das ORIGINALE (volle) Bild anwenden
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated, angle
    
# --- APP KONFIGURATION ---
st.set_page_config(page_title="Präzisions-Analyse Pro", layout="centered")
st.title("🛠 Profil-Mess-App Pro")

# --- SEITENLEISTE ---

st.sidebar.header("Configuration")
orientierung = st.sidebar.radio("Component position:", ("Horizontal (Lying)", "Vertical (Standing)"))
kanten_sens = st.sidebar.slider("edge sensitivity", 0.01, 0.50, 0.14, 0.01)
ref_weiss_mm = st.sidebar.number_input("Reference width inside (mm)", value=60.00)
#such_offset_px_val = st.sidebar.slider("Search offset (pixels)", 1, 100, 30)
such_offset_mm = st.sidebar.slider("Such-Offset (mm)", 0.5, 20.0, 5.0, 0.5)
mm_pro_drehung = st.sidebar.number_input("mm per revolution", value=0.75)
do_auto_level = st.sidebar.checkbox("Bilder automatisch begradigen", value=True)
st.sidebar.markdown("---")
manual_angle = st.sidebar.slider("Manuelle Fein-Drehung (°)", -5.0, 5.0, 0.0, 0.05)

# --- BILD-EINGABE ---
input_method = st.radio("Image source:", ("Use camera", "Screenshot / Upload file"))
uploaded_file = st.camera_input("Foto") if input_method == "Use camera" else st.file_uploader("Select image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Vorbereitung
    pil_img = ImageOps.exif_transpose(Image.open(uploaded_file))
    if pil_img.width > 2000:
        ratio = 2000 / float(pil_img.width)
        pil_img = pil_img.resize((2000, int(pil_img.height * ratio)), Image.Resampling.LANCZOS)
    
    img_rgb = np.array(pil_img.convert('RGB'))
    img_rot = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE) if orientierung == "Horizontal (Lying)" else img_rgb

    # ... (nach der Zeile img_rot = cv2.rotate...)
    if do_auto_level:
        img_rot, gefundener_winkel = auto_begradigen(img_rot)
        st.sidebar.info(f"Korrektur-Winkel: {gefundener_winkel:.2f}°")
   # Dann die manuelle Fein-Justierung oben drauf
    if manual_angle != 0:
        (h, w) = img_rot.shape[:2]
        M_manual = cv2.getRotationMatrix2D((w // 2, h // 2), manual_angle, 1.0)
        img_rot = cv2.warpAffine(img_rot, M_manual, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    st.sidebar.caption(f"Korrektur Gesamt: {auto_angle + manual_angle:.2f}°")     

    # 2. Analyse
    gray = (0.2989 * img_rot[:,:,0] + 0.5870 * img_rot[:,:,1] + 0.1140 * img_rot[:,:,2]).astype(np.float64)
    gray_smooth = convolve2d(gray, np.ones((3, 3))/25.0, mode='same')
    h_grad = np.abs(np.diff(gray_smooth, axis=1))
    kanten_profil = np.mean(h_grad, axis=0)
    kanten_profil = kanten_profil / np.max(kanten_profil)

    # 3. Kanten finden (Mitte -> Außen)
    img_center = kanten_profil.shape[0] // 2
    # Innenkanten (Grün)
    suche_r = np.where(kanten_profil[img_center:] > kanten_sens)[0]
    x_rechts_w_px = (img_center + suche_r[0]) if len(suche_r) > 0 else img_center
    suche_l = np.where(kanten_profil[:img_center][::-1] > kanten_sens)[0]
    x_links_w_px = (img_center - suche_l[0]) if len(suche_l) > 0 else img_center

    if x_rechts_w_px > x_links_w_px:
        px_pro_mm = (x_rechts_w_px - x_links_w_px) / ref_weiss_mm

        such_offset_px_val = int(such_offset_mm * px_pro_mm)
        # Außenkanten (Gelb)
        start_r_a = min(len(kanten_profil)-1, x_rechts_w_px + such_offset_px_val)
        idx_r_a = np.where(kanten_profil[x_rechts_w_px+5:start_r_a+1][::-1] > kanten_sens)[0]
        x_rechts_a_px = (start_r_a - idx_r_a[0]) if len(idx_r_a) > 0 else start_r_a

        start_l_a = max(0, x_links_w_px - such_offset_px_val)
        idx_l_a = np.where(kanten_profil[start_l_a:x_links_w_px-5] > kanten_sens)[0]
        x_links_a_px = (start_l_a + idx_l_a[0]) if len(idx_l_a) > 0 else start_l_a

        # 4. Berechnung
        zentrum_ist_px = (x_links_w_px + x_rechts_w_px) / 2.0
        zentrum_soll_px = (x_links_a_px + x_rechts_a_px) / 2.0
        abweichung_mm = (zentrum_ist_px - zentrum_soll_px) / px_pro_mm
        toleranz_mm = berechne_mess_toleranz(px_pro_mm)
        raw_umdr = abs(abweichung_mm) / mm_pro_drehung
        umdrehungen = round(raw_umdr * 4) / 4  # Rundet auf 0, 0.25, 0.5, 0.75, 1.0 etc.
        
        anweisung = "RIGHT" if abweichung_mm <= 0 else "LEFT"

        # --- ERGEBNIS ANZEIGE ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Deviation", f"{abs(abweichung_mm):.2f} mm")
        col2.metric("Tolerance (±)", f"{toleranz_mm:.2f} mm")
        col3.metric("Correction", f"{umdrehungen} Umdr.")

        if umdrehungen < 0.25:
            st.info(f"✅ Centered (Deviation < 0.2 mm)")
        else:
            st.success(f"⚙️ Turn the Screw **{umdrehungen}** to the **{anweisung}**.")

        # --- VISUALISIERUNG ---
        img_marked = img_rot.copy()
        h_img = img_marked.shape[0]
        for x, c, w in [(x_links_a_px, (255,255,0), 2), (x_rechts_a_px, (255,255,0), 2), 
                        (x_links_w_px, (0,255,0), 2), (x_rechts_w_px, (0,255,0), 2),
                        (zentrum_soll_px, (0,0,255), 2), (zentrum_ist_px, (255,0,0), 2)]:
            cv2.line(img_marked, (int(x), 0), (int(x), h_img), c, w)

        st.subheader("🔍 Detail-Zoom")
        z_cols = st.columns(2)
        y_m = h_img // 2
        def zoom(img, x, y):
            x1, y1, x2, y2 = max(0,int(x-75)), max(0,int(y-75)), min(img.shape[1],int(x+75)), min(img.shape[0],int(y+75))
            return cv2.resize(img[y1:y2, x1:x2], (0,0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
        
        z_cols[0].image(zoom(img_marked, x_links_a_px, y_m), caption="Left Edge")
        z_cols[1].image(zoom(img_marked, x_rechts_a_px, y_m), caption="Right Edge")
        
        st.subheader("Analysis overview")
        st.image(img_marked, use_container_width=True)

        # --- DIAGRAMM UNTER DER ABBILDUNG ---
        st.divider()
        st.subheader("📊 Kanten-Signal-Analyse")
        
        fig, ax = plt.subplots(figsize=(10, 3))
        # Plot des Profils
        ax.plot(kanten_profil, color='cyan', label='Kontrast-Stärke')
        # Schwellenwert-Linie
        ax.axhline(y=kanten_sens, color='red', linestyle='--', label='Schwelle')
        
        # Positionen der gefundenen Kanten markieren
        ax.axvline(x=x_links_w_px, color='green', alpha=0.5, label='Innen')
        ax.axvline(x=x_rechts_w_px, color='green', alpha=0.5)
        ax.axvline(x=x_links_a_px, color='orange', alpha=0.5, label='Außen')
        ax.axvline(x=x_rechts_a_px, color='orange', alpha=0.5)

        ax.set_ylim(0, 1.1)
        ax.set_title("Verlauf der Kantenstärken (Peaks)")
        ax.legend(loc='upper right')
        st.pyplot(fig)
        # --- GRADIENTEN-DIAGRAMM ---
        st.divider()
        st.subheader("📊 Signal-Analyse: Kanten-Profil")
        st.write("Dieses Diagramm zeigt die Kontraststärke (Gradient) über die Bildbreite. Die Spitzen (Peaks) sind deine Kanten.")
        
        # Erstellung des Plots
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Hintergrund und Stil an Streamlit anpassen
        fig.patch.set_facecolor('#0E1117') 
        ax.set_facecolor('#1e2129')
        
        # Den Gradienten (das Kantenprofil) plotten
        ax.plot(kanten_profil, color='#00d1ff', linewidth=1.5, label='Kontrast-Stärke')
        ax.fill_between(range(len(kanten_profil)), kanten_profil, color='#00d1ff', alpha=0.1)
        
        # Die Sensitivitäts-Schwelle einzeichnen
        ax.axhline(y=kanten_sens, color='red', linestyle=':', label='Schwellenwert (Sens)')
        
        # Vertikale Linien für die gefundenen Kantenpositionen
        ax.axvline(x=x_links_w_px, color='lime', linestyle='--', alpha=0.8, label='Innen-Kanten')
        ax.axvline(x=x_rechts_w_px, color='lime', linestyle='--', alpha=0.8)
        
        ax.axvline(x=x_links_a_px, color='yellow', linestyle='--', alpha=0.8, label='Außen-Kanten')
        ax.axvline(x=x_rechts_a_px, color='yellow', linestyle='--', alpha=0.8)

        # Achsen-Beschriftung
        ax.set_xlim(0, len(kanten_profil))
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Pixel-Position (X)", color='white')
        ax.set_ylabel("Relative Stärke", color='white')
        ax.tick_params(colors='white')
        
        # Legende anzeigen
        ax.legend(loc='upper right', facecolor='#0E1117', labelcolor='white', fontsize='small')
        
        # Plot in Streamlit ausgeben
        st.pyplot(fig)


    
    else:

        st.error("Could not find any edges.")










