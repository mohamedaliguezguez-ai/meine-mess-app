import streamlit as st
import numpy as np
import cv2
from scipy.signal import convolve2d
from PIL import Image, ImageOps

# --- HILFSFUNKTIONEN ---
def berechne_mess_toleranz(px_pro_mm, unsicherheit_px=1.5):
    """Berechnet die physikalische Toleranz der Messung in mm."""
    if px_pro_mm > 0:
        return unsicherheit_px / px_pro_mm
    return 0.0

def get_edge_line(grad_img, rows, x_approx, search_range, sensitivity, direction="right"):
    """
    Sucht in mehreren Zeilen nach dem Kontrast-Maximum (Peak) und 
    berechnet eine Ausgleichsgerade (Regression).
    """
    pts_x = []
    pts_y = []
    
    # Globales Maximum für relative Sensitivität
    max_grad = np.max(grad_img)
    if max_grad == 0: return None
    
    for y in rows:
        # Suchfenster definieren
        if direction == "right":
            x1 = int(x_approx + 2) # Kleiner Puffer nach innen
            x2 = int(x_approx + search_range + 30) # Großzügiger Puffer nach außen
        else:
            x1 = int(x_approx - search_range - 30)
            x2 = int(x_approx - 2)
        
        # Grenzprüfung
        x1, x2 = max(0, x1), min(grad_img.shape[1], x2)
        if x1 >= x2: continue
        
        row_segment = grad_img[y, x1:x2]
        if len(row_segment) == 0: continue
        
        rel_idx = np.argmax(row_segment)
        # Validierung des Peaks
        # Außenkanten sind oft schwächer, daher 0.7 Korrekturfaktor
        limit = sensitivity * max_grad * (1.0 if "inner" in direction else 0.7)
        
        if row_segment[rel_idx] > limit:
            pts_x.append(x1 + rel_idx)
            pts_y.append(y)
            
    if len(pts_x) > 8: # Mindestens 8 Punkte für eine stabile Gerade
        m, b = np.polyfit(pts_y, pts_x, 1)
        return m, b, pts_x, pts_y
    return None

# --- APP KONFIGURATION ---
st.set_page_config(page_title="Präzisions-Analyse Pro", layout="centered")
st.title("🛠 Profil-Mess-App: Zeilen-Analyse")

# --- SEITENLEISTE ---
st.sidebar.header("Konfiguration")
orientierung = st.sidebar.radio("Bauteil-Lage:", ("Horizontal (Liegend)", "Vertikal (Stehend)"))
kanten_sens = st.sidebar.slider("Kanten-Sensitivität", 0.01, 0.90, 0.25, 0.05)
ref_weiss_mm = st.sidebar.number_input("Referenzbreite Innen (mm)", value=60.00)
such_offset_mm = st.sidebar.slider("Such-Bereich Außen (mm)", 1.0, 30.0, 8.0, 0.5)
mm_pro_drehung = st.sidebar.number_input("mm pro Schraubendrehung", value=0.75)

# --- BILD-EINGABE ---
input_method = st.radio("Bildquelle:", ("Kamera nutzen", "Datei hochladen"))
uploaded_file = st.camera_input("Foto") if input_method == "Kamera nutzen" else st.file_uploader("Bild auswählen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Vorbereitung
    pil_img = ImageOps.exif_transpose(Image.open(uploaded_file))
    # Bildgröße für Performance begrenzen
    if pil_img.width > 1600:
        ratio = 1600 / float(pil_img.width)
        pil_img = pil_img.resize((1600, int(pil_img.height * ratio)), Image.Resampling.LANCZOS)
    
    img_rgb = np.array(pil_img.convert('RGB'))
    img_rot = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE) if orientierung == "Horizontal (Liegend)" else img_rgb
    h_img, w_img = img_rot.shape[:2]

    # 2. Analyse (Gradientenbild erstellen)
    gray = cv2.cvtColor(img_rot, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gray_smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    # Sobel-Operator für präzise Kantenstärken
    h_grad = np.abs(cv2.Sobel(gray_smooth, cv2.CV_64F, 1, 0, ksize=3))

    # Mess-Zeilen definieren (gleichmäßig über das Bild verteilt)
    sample_rows = np.linspace(h_img * 0.15, h_img * 0.85, 40, dtype=int)
    img_center_x = w_img // 2

    # 3. Kanten finden per Regression
    # Innen Rechts (Suche von Mitte nach rechts)
    res_ir = get_edge_line(h_grad, sample_rows, img_center_x, w_img//2, kanten_sens, direction="right_inner")
    # Innen Links (Suche von Mitte nach links)
    res_il = get_edge_line(h_grad, sample_rows, img_center_x, w_img//2, kanten_sens, direction="left_inner")

    if res_ir and res_il:
        m_ir, b_ir, pts_ir_x, pts_ir_y = res_ir
        m_il, b_il, pts_il_x, pts_il_y = res_il
        
        # Zentrale Berechnungsebene (Bildmitte)
        y_mid = h_img // 2
        x_ir_mid = m_ir * y_mid + b_ir
        x_il_mid = m_il * y_mid + b_il
        
        # Pixel-Skalierung berechnen
        px_pro_mm = (x_ir_mid - x_il_mid) / ref_weiss_mm
        offset_px = such_offset_mm * px_pro_mm

        # 4. Außenkanten finden (Gelbe Linien)
        res_ar = get_edge_line(h_grad, sample_rows, x_ir_mid, offset_px, kanten_sens, direction="right")
        res_al = get_edge_line(h_grad, sample_rows, x_il_mid, offset_px, kanten_sens, direction="left")

        if res_ar and res_al:
            m_ar, b_ar, _, _ = res_ar
            m_al, b_al, _, _ = res_al
            
            x_ar_mid = m_ar * y_mid + b_ar
            x_al_mid = m_al * y_mid + b_al
            
            # 5. Berechnung der Ergebnisse
            zentrum_ist = (x_il_mid + x_ir_mid) / 2.0
            zentrum_soll = (x_al_mid + x_ar_mid) / 2.0
            abweichung_mm = (zentrum_ist - zentrum_soll) / px_pro_mm
            
            winkel = np.degrees(np.arctan((m_ir + m_il + m_ar + m_al) / 4))
            
            # --- ERGEBNIS ANZEIGE ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Abweichung", f"{abs(abweichung_mm):.2f} mm")
            col2.metric("Toleranz (±)", f"{berechne_mess_toleranz(px_pro_mm):.2f} mm")
            raw_umdr = abs(abweichung_mm) / mm_pro_drehung
            umdrehungen = round(raw_umdr * 4) / 4
            col3.metric("Korrektur", f"{umdrehungen} Umdr.")

            if umdrehungen < 0.25:
                st.info("✅ Bauteil ist korrekt zentriert.")
            else:
                richtung = "LINKS" if abweichung_mm > 0 else "RECHTS"
                st.success(f"⚙️ Schraube **{umdrehungen}** Umdrehung(en) nach **{richtung}** drehen.")
            
            st.write(f"📐 Winkel-Korrektur: {winkel:.1f}°")

            # --- VISUALISIERUNG ---
            img_marked = img_rot.copy()
            # Zeichne die vier Haupt-Geraden
            for (m, b, color) in [(m_al, b_al, (255, 255, 0)), (m_ar, b_ar, (255, 255, 0)), 
                                  (m_il, b_il, (0, 255, 0)), (m_ir, b_ir, (0, 255, 0))]:
                x0 = int(m * 0 + b)
                xH = int(m * h_img + b)
                cv2.line(img_marked, (x0, 0), (xH, h_img), color, 4)

            # Zentren (Soll=Blau, Ist=Rot)
            cv2.line(img_marked, (int(zentrum_soll), 0), (int(zentrum_soll), h_img), (0, 0, 255), 2)
            cv2.line(img_marked, (int(zentrum_ist), 0), (int(zentrum_ist), h_img), (255, 0, 0), 2)

            st.image(img_marked, caption="Analyse-Ergebnis (Grün: Innen, Gelb: Außen)", use_container_width=True)
            
        else:
            st.error("Außenkanten (Gelb) nicht gefunden. Versuchen Sie den 'Such-Bereich Außen' zu vergrößern.")
            st.image(img_rot, use_container_width=True)
    else:
        st.error("Innenkanten (Grün) nicht gefunden. Bitte Sensitivität oder Bildposition prüfen.")
