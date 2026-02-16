def get_edge_line(grad_img, rows, x_approx, search_range, sensitivity, direction="right"):
    """
    Sucht nach der Kante in einem Puffer-Bereich um einen geschätzten Punkt (x_approx).
    """
    pts_x = []
    pts_y = []
    
    # Globales Maximum für relative Sensitivität (etwas herabgesetzt für Außenkanten)
    max_grad = np.max(grad_img)
    
    for y in rows:
        if direction == "right":
            # Suche rechts von der Innenkante
            x1 = int(x_approx + 5) 
            x2 = int(x_approx + search_range + 20) # 20px Puffer dazu
        else:
            # Suche links von der Innenkante
            x1 = int(x_approx - search_range - 20) # 20px Puffer dazu
            x2 = int(x_approx - 5)
        
        # Sicherstellen, dass wir innerhalb des Bildes bleiben
        x1, x2 = max(0, x1), min(grad_img.shape[1], x2)
        if x1 >= x2: continue
        
        row_segment = grad_img[y, x1:x2]
        if len(row_segment) == 0: continue
        
        rel_idx = np.argmax(row_segment)
        # Wir nehmen den Peak, wenn er eine Mindeststärke hat
        if row_segment[rel_idx] > (sensitivity * max_grad * 0.7): # 0.7 als Bonus für schwächere Außenkanten
            pts_x.append(x1 + rel_idx)
            pts_y.append(y)
            
    if len(pts_x) > 5:
        m, b = np.polyfit(pts_y, pts_x, 1)
        return m, b
    return None

# --- DER AUFRUF IM HAUPTCODE ---
if line_ir and line_il:
    # ... (Berechnung von px_pro_mm wie zuvor) ...
    
    # Gelbe Linien (Außenkanten) mit der neuen Funktion suchen:
    # Rechts außen: Startet bei der grünen Linie, sucht nach rechts
    line_ar = get_edge_line(h_grad, sample_rows, x_ir_mid, offset_px, kanten_sens, direction="right")
    
    # Links außen: Startet bei der grünen Linie, sucht nach links
    line_al = get_edge_line(h_grad, sample_rows, x_il_mid, offset_px, kanten_sens, direction="left")

    if line_ar and line_al:
        # ... (Rest der Berechnung) ...
