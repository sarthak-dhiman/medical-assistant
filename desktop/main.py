import webview
import sys
import os
import threading
from inference import predict_image # We will create this next

class Api:
    def predict(self, image_data, mode, debug=False):
        """
        Bridge method called from React: window.pywebview.api.predict(img, mode, debug)
        """
        try:
            # Run inference synchronously (ONNX on CPU is fast enough for UI)
            result = predict_image(image_data, mode, debug)
            return {"status": "success", "mode": mode, **result}
        except Exception as e:
            return {"status": "error", "error": str(e)}

def get_entrypoint():
    """
    Returns the URL or file path for the frontend.
    - Frozen (Exe): Returns path to bundled index.html
    - Dev: Returns localhost:5173 (Vite dev server)
    """
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller Bundle
        base_dir = sys._MEIPASS
        html_path = os.path.join(base_dir, 'dist', 'index.html')
        if not os.path.exists(html_path):
             return f"data:text/html,<html><body><h1>Error: Dist Not Found at {html_path}</h1></body></html>"
        return f"file://{html_path}"
    else:
        # Running as Script (Development)
        # Check if local dist exists, else assume dev server
        local_dist = os.path.join(os.path.dirname(__file__), '..', 'web_app', 'frontend', 'dist', 'index.html')
        if os.path.exists(local_dist):
            return f"file://{os.path.abspath(local_dist)}"
        return 'http://localhost:5173'

if __name__ == '__main__':
    api = Api()
    window = webview.create_window(
        'Medical AI Assistant (Desktop)', 
        url=get_entrypoint(), 
        js_api=api, 
        width=1280, 
        height=800,
        resizable=True
    )
    webview.start(debug=True)
