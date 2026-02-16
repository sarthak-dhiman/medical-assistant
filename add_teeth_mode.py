import re

# Read the file
with open(r'd:\Disease Prediction\webcam_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define the teeth mode code
teeth_mode_code = '''
        # --- MODE 6: TEETH ---
        elif current_model == "TEETH":
            # Teeth detection works on oral/mouth images - use center region or full frame
            if cv2.countNonZero(final_skin_mask) > (threshold/2):
                masked_roi = cv2.bitwise_and(frame, frame, mask=final_skin_mask)
                y_indices, x_indices = np.where(final_skin_mask > 0)
                if len(y_indices) > 0:
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    cropped_skin = masked_roi[y_min:y_max, x_min:x_max]
                    if cropped_skin.size > 0:
                        label, conf, _ = predict_teeth_disease(cropped_skin, debug=False)
                        color = (0, 255, 0) if "Healthy" in label else (0, 0, 255)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                        sidebar_result = label
                        sidebar_conf = f"Conf: {conf*100:.1f}%"
            else:
                sidebar_warning = "Show Teeth/Mouth"
                cy, cx = h//2, w//2
                crop_h, crop_w = h//2, w//2
                y1, y2 = max(0, cy - crop_h//2), min(h, cy + crop_h//2)
                x1, x2 = max(0, cx - crop_w//2), min(w, cx + crop_w//2)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    label, conf, _ = predict_teeth_disease(crop, debug=False)
                    color = (0, 255, 0) if "Healthy" in label else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    sidebar_result = label
                    sidebar_conf = f"Conf: {conf*100:.1f}%"

'''

# Find the insertion point (after burns mode, before "DRAW FINAL UI")
pattern = r'(                cv2\.rectangle\(frame, \(cx-100, cy-100\), \(cx\+100, cy\+100\), \(255, 255, 0\), 2\)\r?\n)\r?\n\r?\n(        # --- DRAW FINAL UI ---)'

# Replace with the pattern + teeth mode + draw final ui
replacement = r'\1' + teeth_mode_code + r'\2'

new_content = re.sub(pattern, replacement, content)

# Write back
with open(r'd:\Disease Prediction\webcam_app.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Teeth mode added successfully!")
