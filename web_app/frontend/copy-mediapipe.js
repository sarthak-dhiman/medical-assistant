import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const srcDir = path.resolve(__dirname, 'node_modules', '@mediapipe', 'face_mesh');
const destDir = path.resolve(__dirname, 'public', 'mediapipe', 'face_mesh');

try {
    fs.mkdirSync(destDir, { recursive: true });

    if (fs.existsSync(srcDir)) {
        const files = fs.readdirSync(srcDir);
        files.forEach(file => {
            if (
                file.endsWith('.js') ||
                file.endsWith('.wasm') ||
                file.endsWith('.data') ||
                file.endsWith('.binarypb') ||
                file.endsWith('.tflite')
            ) {
                fs.copyFileSync(path.join(srcDir, file), path.join(destDir, file));
            }
        });
        console.log('Successfully copied MediaPipe assets to public/mediapipe/face_mesh');
    } else {
        console.warn(`MediaPipe source directory not found: ${srcDir}`);
    }
} catch (error) {
    console.error('Error copying MediaPipe assets:', error);
}
