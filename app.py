import streamlit as st
import numpy as np
import cv2
from scipy.signal import convolve2d
from PIL import Image, ImageOps

# --- HILFSFUNKTIONEN ---
def berechne_mess_toleranz(px_pro_mm, unsicherheit_px=1.5):
    if px_pro_mm > 0:
        return unsicherheit_px / px_pro_mm
    return 0.0

def get_edge_line(grad_img, rows, x_start, x_end, sensitivity):
    """Sucht in den angegebenen Zeilen nach dem Peak und fittet eine Gerade x = m*y + b"""
    pts_x = []
    pts_y = []
    
    # Globales Maximum für relative Sensitivität
    max_grad = np.max(grad_img)
    
    for y in rows:
        # Bereich extrahieren (Sicherstellen, dass Indizes innerhalb des Bildes liegen)
        x1, x2 = max(0, int(x_start)), min(grad_img.shape[1], int(x_end))
        if x1 >= x2: continue
        
        row_segment = grad_img[y, x1:x2]
        if len(row_segment) == 0: continue
        
        rel_idx = np.argmax(row_segment)
        if row_segment[rel_idx] > (sensitivity * max_grad):
            pts_x.append(x1 + rel_idx)
            pts_y.append(y)
            
    if len(pts_x) > 5: # Mindestens 5 Punkte für eine stabile Gerade
        m, b = np.polyfit(pts_y, pts_x, 1)
        return m, b
    return None

# --- APP KONFIGURATION ---
st.set_page_config(page_title="Präzisions-Analyse Pro (Line-by-Line)", layout="centered")
st.title("🛠 Profil-Mess-App: Zeilen-Analyse")

# --- SEITENLEISTE ---
st.sidebar.header("Konfiguration")
orientierung = st.sidebar.radio("Bauteil-Lage:", ("Horizontal (Liegend)", "Vertikal (Stehend)"))
kanten_sens = st.sidebar.slider("Kanten-Sensitivität", 0.01, 0.90, 0.30, 0.05)
ref_weiss_mm = st.sidebar.number_input("Referenzbreite Innen (mm)", value=60.00)
such_offset_mm = st.sidebar.slider("Such-Offset Außen (mm)", 0.5, 20.0, 5.0, 0.5)
mm_pro_drehung = st.sidebar.number_input("mm pro Schraubendrehung", value=0.75)

# --- BILD-EINGABE ---
input_method = st.radio("Bildquelle:", ("Kamera nutzen", "Datei hochladen"))
uploaded_file = st.camera_input("Foto") if input_method == "Kamera nutzen" else st.file_uploader("Bild auswählen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Vorbereitung
    pil_img = ImageOps.exif_transpose(Image.open(uploaded_file))
    if pil_img.width > 1500:
        ratio = 1500 / float(pil_img.width)
        pil_img = pil_img.resize((1500, int(pil_img.height * ratio)), Image.Resampling.LANCZOS)
    
    img_rgb = np.array(pil_img.convert('RGB'))
    img_rot = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE) if orientierung == "Horizontal (Liegend)" else img_rgb
    h_img, w_img = img_rot.shape[:2]

    # 2. Analyse (Gradientenbild erstellen)
    gray = cv2.cvtColor(img_rot, cv2.COLOR_RGB2GRAY).astype(np.float64)
    # Kleinerer Kernel für schärfere Peaks
    gray_smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    h_grad = np.abs(cv2.Sobel(gray_smooth, cv2.CV_64F, 1, 0, ksize=3))

    # Zeilen für die Analyse auswählen (z.B. 30 Zeilen über die Höhe verteilt)
    sample_rows = np.linspace(h_img*0.1, h_img*0.9, 30, dtype=int)
    img_center_x = w_img // 2

    # 3. Kanten finden per Regression
    # Innen Rechts (Suche von Mitte nach rechts)
    line_ir = get_edge_line(h_grad, sample_rows, img_center_x, w_img, kanten_sens)
    # Innen Links (Suche von Mitte nach links)
    line_il = get_edge_line(h_grad, sample_rows, 0, img_center_x, kanten_sens)

    if line_ir and line_il:
        # Funktionen zur X-Berechnung: x = m*y + b
        m_ir, b_ir = line_ir
        m_il, b_il = line_il
        
        # Breite in der Bildmitte berechnen (y_mid)
        y_mid = h_img // 2
        x_ir_mid = m_ir * y_mid + b_ir
        x_il_mid = m_il * y_mid + b_il
        
        px_pro_mm = (x_ir_mid - x_il_mid) / ref_weiss_mm
        offset_px = such_offset_mm * px_pro_mm

        # Außenkanten (basierend auf Innenkanten + Offset)
        line_ar = get_edge_line(h_grad, sample_rows, x_ir_mid + 5, x_ir_mid + offset_px, kanten_sens)
        line_al = get_edge_line(h_grad, sample_rows, x_il_mid - offset_px, x_il_mid - 5, kanten_sens)

        if line_ar and line_al:
            m_ar, b_ar = line_ar
            m_al, b_al = line_al
            
            # Alle X-Positionen für die Bildmitte berechnen
            x_ar_mid = m_ar * y_mid + b_ar
            x_al_mid = m_al * y_mid + b_al
            
            # 4. Berechnung der Abweichung (in der Bildmitte)
            zentrum_ist = (x_il_mid + x_ir_mid) / 2.0
            zentrum_soll = (x_al_mid + x_ar_mid) / 2.0
            abweichung_mm = (zentrum_ist - zentrum_soll) / px_pro_mm
            
            # Winkel berechnen (nur zur Info)
            winkel = np.degrees(np.arctan(m_ir))
            
            # --- ERGEBNIS ANZEIGE ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Abweichung", f"{abs(abweichung_mm):.2f} mm")
            col2.metric("Toleranz (±)", f"{berechne_mess_toleranz(px_pro_mm):.2f} mm")
            raw_umdr = abs(abweichung_mm) / mm_pro_drehung
            umdrehungen = round(raw_umdr * 4) / 4
            col3.metric("Korrektur", f"{umdrehungen} Umdr.")

            st.write(f"📐 Bauteil-Winkel: {winkel:.1f}°")

            # --- VISUALISIERUNG ---
            img_marked = img_rot.copy()
            for (m, b, color) in [(m_al, b_al, (255,255,0)), (m_ar, b_ar, (255,255,0)), 
                                  (m_il, b_il, (0,255,0)), (m_ir, b_ir, (0,255,0))]:
                # Zeichne Linie von oben nach unten
                cv2.line(img_marked, (int(m*0 + b), 0), (int(m*h_img + b), h_img), color, 3)

            # Zentren markieren
            cv2.line(img_marked, (int(zentrum_soll), 0), (int(zentrum_soll), h_img), (0,0,255), 2)
            cv2.line(img_marked, (int(zentrum_ist), 0), (int(zentrum_ist), h_img), (255,0,0), 2)

            st.image(img_marked, caption="Analyse mit Regressionsgeraden", use_container_width=True)
        else:
            st.error("Außenkanten konnten nicht präzise berechnet werden.")
    else:
        st.error("Innenkanten nicht gefunden. Bitte Sensitivität anpassen.")
