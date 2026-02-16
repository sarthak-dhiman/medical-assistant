import re

# Read the file
with open(r'd:\Disease Prediction\web_app\backend\tasks.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define the teeth mode routing code
teeth_routing = '''        elif mode == "TEETH":
            # Teeth disease is a direct classification on teeth/mouth images
            try:
                label, conf, debug_info = inference_service.predict_teeth_disease(frame, debug=debug)
                result = {"status": "success", "mode": mode, "label": label, "confidence": float(conf), "debug_info": debug_info}
                recommendations = inference_service.get_recommendations(label)
                if recommendations:
                    result["recommendations"] = recommendations
                return result
            except Exception as e:
                logger.error(f"Teeth disease processing error: {e}")
                return {"status": "error", "error": f"Teeth disease detection failed: {str(e)}"}
        '''

# Find and replace the pattern
pattern = r'(        elif mode in \["JAUNDICE_BODY", "JAUNDICE_EYE", "SKIN_DISEASE", "BURNS", "NAIL_DISEASE"\]:)'
replacement = teeth_routing + r'\1'

if 'elif mode == "TEETH":' not in content:
    new_content = re.sub(pattern, replacement, content)
    
    # Write back
    with open(r'd:\Disease Prediction\web_app\backend\tasks.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("Teeth mode routing added to tasks.py successfully!")
else:
    print("Teeth mode routing already exists in tasks.py")
