import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# --- APP KONFIGURATION ---
st.set_page_config(page_title="Präzisions-Analyse Pro", layout="centered")

st.title("🛠 Profil-Mess-App Pro")
st.write("Nutze die Seitenleiste, um die Empfindlichkeit anzupassen.")

# --- SEITENLEISTE (EINSTELLUNGEN) ---
st.sidebar.header("Einstellungen")

kanten_sens = st.sidebar.slider(
    "Kanten-Sensibilität", 
    min_value=0.01, 
    max_value=0.30, 
    value=0.05, 
    step=0.01,
    help="Niedriger Wert = erkennt feinste Kanten. Hoher Wert = ignoriert Schatten."
)

ref_weiss_mm = st.sidebar.number_input("Referenzbreite Weiß (mm)", value=90.0)
such_offset_mm = st.sidebar.slider("Such-Offset (mm)", 5, 30, 10)
mm_pro_drehung = st.sidebar.number_input("mm pro Umdrehung", value=0.75)

# --- BILD-UPLOAD ---
uploaded_file = st.sidebar.file_uploader("Bild auswählen oder Foto machen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild laden
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    
    # Verarbeitung
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rot = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)
    
    # Graustufenwandlung (MATLAB-Gewichte)
    gray = (0.2989 * img_rot[:,:,0] + 0.5870 * img_rot[:,:,1] + 0.1140 * img_rot[:,:,2]).astype(np.float64)

    # Kanten-Profil erstellen
    kernel = np.ones((5, 5), np.float64) / 25.0
    gray_smooth = convolve2d(gray, kernel, mode='same')
    h_grad = np.abs(np.diff(gray_smooth, axis=1))
    h_grad = np.hstack((h_grad, np.zeros((h_grad.shape[0], 1))))
    kanten_profil = np.mean(h_grad, axis=0)
    kanten_profil = kanten_profil / np.max(kanten_profil)

    # Innenkanten (Grün) finden
    bw_weiss = gray > 180
    x_proj_w = np.sum(bw_weiss, axis=0)
    weiss_idx = np.where(x_proj_w > (np.max(x_proj_w) * 0.5))[0]
    
    if len(weiss_idx) > 0:
        x_links_w_px = weiss_idx[0]
        x_rechts_w_px = weiss_idx[-1]
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

        # --- ANZEIGE DER ERGEBNISSE ---
        col1, col2 = st.columns(2)
        col1.metric("Abweichung", f"{abs(abweichung_mm):.2f} mm")
        col2.metric("Korrektur", f"{umdrehungen:.2f} Umdr.")
        
        st.info(f"👉 Bitte die Schraube **{anweisung}** drehen.")

        # --- ZOOM-LOGIK (5x Vergrößerung mit Pfeilen) ---
        st.subheader("🔍 Detail-Ansicht Kanten (5x Zoom)")
        z_cols = st.columns(2)
        y_mid = img_rot.shape[0] // 2 

        def draw_zoom(img, x_center, y_center, size, scale, marks):
            x1 = max(0, x_center - size // 2)
            y1 = max(0, y_center - size // 2)
            x2 = min(img.shape[1], x_center + size // 2)
            y2 = min(img.shape[0], y_center + size // 2)
            
            crop = img[y1:y2, x1:x2].copy()
            resized = cv2.resize(crop, (None, None), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            
            for x_px, color in marks:
                rel_x = int((x_px - x1) * scale)
                if 0 <= rel_x < resized.shape[1]:
                    # Zeichne Pfeil (Start, Ende, Farbe, Dicke)
                    cv2.arrowedLine(resized, (rel_x, 15), (rel_x, 65), color, 4, tipLength=0.3)
            return resized

        zoom_f = 5
        zoom_s = 80 

        # Linker Zoom (Markierung für Außenkante gelb und Innenkante grün)
        marks_l = [(x_links_a_px, (255, 255, 0)), (x_links_w_px, (0, 255, 0))]
        x_mid_l = (x_links_a_px + x_links_w_px) // 2
        zoom_l = draw_zoom(img_rot, x_mid_l, y_mid, zoom_s, zoom_f, marks_l)
        z_cols[0].image(zoom_l, caption="Links (Gelb=Außen, Grün=Innen)", use_container_width=True)

        # Rechter Zoom (Markierung für Innenkante grün und Außenkante gelb)
        marks_r = [(x_rechts_w_px, (0, 255, 0)), (x_rechts_a_px, (255, 255, 0))]
        x_mid_r = (x_rechts_a_px + x_rechts_w_px) // 2
        zoom_r = draw_zoom(img_rot, x_mid_r, y_mid, zoom_s, zoom_f, marks_r)
        z_cols[1].image(zoom_r, caption="Rechts (Grün=Innen, Gelb=Außen)", use_container_width=True)

        # --- HAUPTGRAFIKEN ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        ax1.imshow(img_rot.astype(np.uint8))
        ax1.axvline(x_links_a_px, color='yellow', linewidth=3, label='Außen (Gelb)')
        ax1.axvline(x_rechts_a_px, color='yellow', linewidth=3)
        ax1.axvline(x_links_w_px, color='green', linewidth=3, label='Innen (Grün)')
        ax1.axvline(x_rechts_w_px, color='green', linewidth=3)
        ax1.axvline(zentrum_soll_px, color='blue', linewidth=2, label='Soll')
        ax1.axvline(zentrum_ist_px, color='red', linewidth=2, linestyle='--', label='Ist')
        ax1.legend(loc='upper right')
        ax1.set_title("Kamera-Analyse")

        x_mm = (np.arange(len(kanten_profil)) - x_links_a_px) * mm_per_px
        ax2.plot(x_mm, kanten_profil, color='black')
        ax2.axhline(kanten_sens, color='red', linestyle='--', label='Schwelle')
        ax2.set_xlabel("Position [mm]")
        ax2.set_ylabel("Kantenstärke")
        ax2.grid(True)
        
        st.pyplot(fig)
    else:
        st.error("Weißer Bereich nicht erkannt. Bitte Ausrichtung prüfen.")