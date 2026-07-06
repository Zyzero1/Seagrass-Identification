from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
import time
import threading
import base64
import os
import sys
import tempfile
from werkzeug.utils import secure_filename
from datetime import datetime
import logging

app = Flask(__name__)
app.config['SECRET_KEY'] = 'seagrassid-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app_state = {
    "selected_model": None,
    "current_image_path": None,
    "last_identification": None,
    "history": [],
}

state_lock = threading.Lock()

TEMP_DIR = tempfile.mkdtemp(prefix="seagrassid_img_")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp'}

def letterbox_image(image, expected_size):
    """Resize image with padding while keeping aspect ratio."""
    ih, iw = image.shape[:2]
    ew, eh = expected_size

    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)

    image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_img = np.zeros((eh, ew, 3), dtype=np.float32)
    pad_img[0:nh, 0:nw, :] = image_resized.astype(np.float32)

    dx = 0
    dy = 0
    return pad_img, scale, dx, dy


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# YOLOv8: 6 species
YOLO_SPECIES = [
    'Cymodocea rotundata',
    'Enhalus acoroides',
    'Halodule uninervis',
    'Halophila ovalis',
    'Syringodium isoetifolium',
    'Thalassia hemprichii'
]

EFFICIENTDET_SPECIES = [
    'Seagrass',                  # class 0
    'Cymodocea rotundata',       # class 1
    'Enhalus acoroides',         # class 2
    'Halodule uninervis',        # class 3
    'Halophila ovalis',          # class 4
    'Syringodium isoetifolium',  # class 5
    'Thalassia hemprichii'       # class 6
]

VALID_SPECIES = [
    'Cymodocea rotundata',
    'Enhalus acoroides',
    'Halodule uninervis',
    'Halophila ovalis',
    'Syringodium isoetifolium',
    'Thalassia hemprichii'
]

SPECIES_COLORS = {
    'Seagrass': (100, 200, 100),                    # #64c864 - Hijau
    'Cymodocea rotundata': (156, 75, 39),           # #274B9C - Biru Tua
    'Enhalus acoroides': (26, 53, 110),             # #6e351a - Cokelat Tua
    'Halodule uninervis': (196, 156, 46),           # #2e9cc4 - Biru Laut
    'Halophila ovalis': (80, 180, 80),              # #50b450 - Hijau
    'Syringodium isoetifolium': (180, 120, 50),     # #3278b4 - Biru
    'Thalassia hemprichii': (156, 100, 200)         # #c8649c - Pink
}

# ============================================================
# LOAD MODEL
# ============================================================
_yolo_model = None
_effdet_model = None
_model_lock = threading.Lock()

def load_yolo():
    global _yolo_model
    with _model_lock:
        if _yolo_model is None:
            try:
                from ultralytics import YOLO
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                _yolo_model = YOLO(os.path.join(BASE_DIR, 'models', 'best_pseudo.pt'))
                logger.info("[YOLO] Model loaded ✓ (6 species)")
            except Exception as e:
                logger.error(f"[YOLO] Failed to load: {e}")
                _yolo_model = None
    return _yolo_model

def load_efficientdet():
    global _effdet_model
    with _model_lock:
        if _effdet_model is None:
            try:
                from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
                from effdet.efficientdet import HeadNet

                config = get_efficientdet_config('tf_efficientdet_d0')
                config.num_classes = 7
                config.image_size = [512, 512]

                net = EfficientDet(config, pretrained_backbone=False)
                net.class_net = HeadNet(config, num_outputs=7)

                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                ckpt = torch.load(
                    os.path.join(BASE_DIR, 'models', 'effdet-pseudo_0.001.ckpt'),
                    map_location='cpu',
                    weights_only=False
                )
                raw_sd = ckpt['state_dict']

                new_sd = {
                    k[len('predictor.model.'):]: v
                    for k, v in raw_sd.items()
                    if k.startswith('predictor.model.')
                }

                missing, unexpected = net.load_state_dict(new_sd, strict=False)
                logger.info(f"[EfficientDet] Missing: {len(missing)} (bias only, OK) | Unexpected: {len(unexpected)}")

                net.eval()
                _effdet_model = DetBenchPredict(net)
                logger.info("[EfficientDet] Ready ✓ (effdet/timm)")

            except Exception as e:
                import traceback
                logger.error(f"[EfficientDet] Failed:\n{traceback.format_exc()}")
                _effdet_model = None
    return _effdet_model

def draw_boxes_on_image(image, detections):
    """Draw bounding boxes on image (skala proporsional & label selalu terlihat penuh)"""
    img_copy = image.copy()
    h, w = img_copy.shape[:2]

    REF_DIM = 1000.0
    scale_factor = max(1.0, max(h, w) / REF_DIM)

    box_thickness = max(2, int(round(3 * scale_factor)))
    font_scale = 0.5 * scale_factor
    text_thickness = max(1, int(round(2 * scale_factor)))
    padding = max(6, int(round(8 * scale_factor)))

    for det in detections:
        bbox = det['bbox']
        label = det['label']
        conf = det['confidence']

        if label == 'Seagrass':
            continue

        x1 = int(np.clip(bbox[0], 0, w - 1))
        y1 = int(np.clip(bbox[1], 0, h - 1))
        x2 = int(np.clip(bbox[2], 1, w))
        y2 = int(np.clip(bbox[3], 1, h))

        if (x2 - x1) < 5 or (y2 - y1) < 5:
            continue

        color = SPECIES_COLORS.get(label, (39, 75, 156))

        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, box_thickness)

        text = f"{label} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX

        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
        label_h = text_h + padding
        label_w = text_w + padding

        if y1 - label_h >= 0:
            label_y1 = y1 - label_h
            label_y2 = y1
        elif y2 + label_h <= h:
            label_y1 = y2
            label_y2 = y2 + label_h
        else:
            label_y1 = y1
            label_y2 = y1 + label_h

        if label_y2 > h:
            label_y2 = h
            label_y1 = h - label_h
        if label_y1 < 0:
            label_y1 = 0
            label_y2 = label_h

        label_x1 = x1
        label_x2 = x1 + label_w
        if label_x2 > w:
            label_x2 = w
            label_x1 = w - label_w
        if label_x1 < 0:
            label_x1 = 0
            label_x2 = label_w

        label_x1 = int(np.clip(label_x1, 0, w))
        label_x2 = int(np.clip(label_x2, 0, w))
        label_y1 = int(np.clip(label_y1, 0, h))
        label_y2 = int(np.clip(label_y2, 0, h))

        cv2.rectangle(img_copy, (label_x1, label_y1), (label_x2, label_y2), color, cv2.FILLED)
        cv2.putText(
            img_copy, text,
            (label_x1 + padding // 2, label_y2 - padding // 2),
            font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA
        )

    return img_copy

def image_to_base64(img_bgr):
    """Convert BGR image to base64 string"""
    _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')

def identify_with_yolo(image_bgr):
    """Identify using YOLOv8 (6 species) - SINKRON 100% DENGAN NOTEBOOK RETRAIN SEL 16"""
    model = load_yolo()
    if model is None:
        return None, {'error': 'YOLO model not loaded', 'total': 0}
    
    t0 = time.perf_counter()
    
    results = model(
        image_bgr, 
        conf=0.35,     
        iou=0.60,       
        imgsz=512,      
        verbose=False
    )
    
    t1 = time.perf_counter()
    inference_time = (t1 - t0) * 1000
    
    boxes = results[0].boxes
    detections = []
    species_counts = {}
    
    if len(boxes) > 0:
        for box in boxes:
            # Ultralytics otomatis mengembalikan koordinat xyxy ke resolusi asli gambar masukan
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls_id < len(YOLO_SPECIES):
                species = YOLO_SPECIES[cls_id]
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'label': species,
                    'confidence': conf
                })
                species_counts[species] = species_counts.get(species, 0) + 1
    
    annotated_bgr = draw_boxes_on_image(image_bgr, detections)
    
    return annotated_bgr, {
        'detections': detections,
        'species_counts': species_counts,
        'total': len(detections),
        'inference_time_ms': round(inference_time, 2),
        'model_used': 'YOLOv8'
    }

def identify_with_efficientdet(image_bgr):
    """Identify using EfficientDet-D0 via effdet/timm (REPARASI FINAL ASPEK RASIO VERTIKAL)"""
    model = load_efficientdet()
    if model is None:
        return None, {'error': 'EfficientDet model not loaded', 'total': 0}

    try:
        orig_h, orig_w = image_bgr.shape[:2]
        input_size = 512

        t0 = time.perf_counter()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_padded, scale, dx, dy = letterbox_image(image_rgb, (input_size, input_size))
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_padded / 255.0 - mean) / std
        
        device = next(model.parameters()).device
        x = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device).float()

        tgt = {
            'img_scale': torch.ones(1).to(device),
            'img_size' : torch.full((1, 2), float(input_size)).to(device)
        }

        
        with torch.no_grad():
            out = model(x, tgt)


        detections = []
        species_counts = {}
        CONF_THRESHOLD = 0.35

        if out is not None and len(out) > 0:
            preds = out[0].cpu().numpy()
            valid_preds = preds[preds[:, 4] > CONF_THRESHOLD]

            for pred in valid_preds:

                x1_raw, y1_raw, x2_raw, y2_raw, score, cls_id = pred[:6]
                cls_id = int(cls_id)
                score = float(score)

                if cls_id == 0 or cls_id >= len(EFFICIENTDET_SPECIES):
                    continue

                x1 = ((float(x1_raw) - dx) / scale) * 1.05
                y1 = ((float(y1_raw) - dy) / scale) * 1.075
                x2 = ((float(x2_raw) - dx) / scale) * 1.075
                y2 = ((float(y2_raw) - dy) / scale) * 1.05

                x1 = np.clip(x1, 0, orig_w)
                y1 = np.clip(y1, 0, orig_h)
                x2 = np.clip(x2, 0, orig_w)
                y2 = np.clip(y2, 0, orig_h)

                species = EFFICIENTDET_SPECIES[cls_id]
                if species in VALID_SPECIES:
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'label': species,
                        'confidence': score
                    })
                    species_counts[species] = species_counts.get(species, 0) + 1

        t1 = time.perf_counter()
        inference_time = (t1 - t0) * 1000
        
        annotated_bgr = draw_boxes_on_image(image_bgr, detections)
        return annotated_bgr, {
            'detections': detections,
            'species_counts': species_counts,
            'total': len(detections),
            'inference_time_ms': round(inference_time, 2),
            'model_used': 'EfficientDet-D0'
        }

    except Exception as e:
        import traceback
        logger.error(f"[EfficientDet] Error:\n{traceback.format_exc()}")
        return None, {'error': str(e), 'total': 0}

# ROUTES

@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')

@app.route('/select-model', methods=['POST'])
def select_model():
    """API to select identification model"""
    model = request.args.get('model', None)
    with state_lock:
        if model in ("yolov8", "efficientdet"):
            app_state["selected_model"] = model
            app_state["last_identification"] = None
    logger.info(f"Model selected: {model}")
    return jsonify({"status": "success", "model": model})

@app.route('/upload-image', methods=['POST'])
def upload_image():
    """API to upload image"""
    if 'image' not in request.files:
        return jsonify({"error": "No image file"}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not allowed_file(image_file.filename):
        return jsonify({"error": "File type not allowed. Use JPG, PNG, JPEG, WEBP"}), 400
    
    with state_lock:
        old_path = app_state.get("current_image_path")
    if old_path and os.path.exists(old_path):
        try:
            os.remove(old_path)
            annotated_old = old_path.replace('.', '_annotated.')
            if os.path.exists(annotated_old):
                os.remove(annotated_old)
        except Exception:
            pass
    
    timestamp = int(time.time())
    filename = secure_filename(f"{timestamp}_{image_file.filename}")
    dest = os.path.join(TEMP_DIR, filename)
    image_file.save(dest)
    
    with state_lock:
        app_state["current_image_path"] = dest
        app_state["last_identification"] = None
    
    logger.info(f"Image uploaded: {dest}")
    return jsonify({"status": "success", "path": dest, "filename": filename})

@app.route('/identify', methods=['POST'])
def identify_image():
    """API to identify uploaded image"""
    with state_lock:
        selected_model = app_state["selected_model"]
        image_path = app_state.get("current_image_path")
    
    if selected_model is None:
        return jsonify({"error": "No model selected"}), 400
    
    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": "No image uploaded"}), 400
    
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return jsonify({"error": "Failed to read image"}), 400
    
    try:
        if selected_model == 'yolov8':
            annotated_image, results = identify_with_yolo(image_bgr)
        else:
            annotated_image, results = identify_with_efficientdet(image_bgr)
    except Exception as e:
        import traceback
        logger.error(f"[IDENTIFY] CRASH saat inferensi:\n{traceback.format_exc()}")
        return jsonify({"error": f"Server crash: {str(e)}"}), 500

    if annotated_image is None:
        return jsonify({"error": "Identification failed", "details": results.get('error', 'Unknown error')}), 500
    
    annotated_path = image_path.replace('.', '_annotated.')
    cv2.imwrite(annotated_path, annotated_image)
    
    annotated_base64 = image_to_base64(annotated_image)
    
    history_entry = {
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'date': datetime.now().strftime("%Y-%m-%d"),
        'model': selected_model,
        'species': results.get('species_counts', {}),
        'total': results.get('total', 0),
        'inference_time_ms': results.get('inference_time_ms', 0)
    }
    
    with state_lock:
        app_state["last_identification"] = results
        app_state["history"].insert(0, history_entry)
        if len(app_state["history"]) > 20:
            app_state["history"] = app_state["history"][:20]
    
    response = {
        "success": True,
        "annotated_image": annotated_base64,
        "detections": results.get('detections', []),
        "species": results.get('species_counts', {}),
        "total": results.get('total', 0),
        "inference_time_ms": results.get('inference_time_ms', 0),
        "model_used": results.get('model_used', selected_model)
    }
    
    logger.info(f"Identification completed: {results.get('total', 0)} objects detected")
    return jsonify(response)

@app.route('/stats', methods=['GET'])
def get_stats():
    """API to get identification statistics"""
    with state_lock:
        history = app_state.get("history", [])
        
        total_all_time = sum(h.get('total', 0) for h in history)
        
        species_all_time = {}
        for h in history:
            for species, count in h.get('species', {}).items():
                species_all_time[species] = species_all_time.get(species, 0) + count
        
        stats = {
            "total_all_time": total_all_time,
            "species_all_time": species_all_time,
            "history_count": len(history)
        }
    return jsonify(stats)

@app.route('/reset', methods=['POST'])
def reset():
    """API to reset state"""
    with state_lock:
        if app_state.get("current_image_path") and os.path.exists(app_state["current_image_path"]):
            try:
                os.remove(app_state["current_image_path"])
                annotated_path = app_state["current_image_path"].replace('.', '_annotated.')
                if os.path.exists(annotated_path):
                    os.remove(annotated_path)
            except Exception:
                pass
        app_state["current_image_path"] = None
        app_state["last_identification"] = None
    return jsonify({"status": "success"})

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """API to clear identification history"""
    with state_lock:
        app_state["history"] = []
    return jsonify({"status": "success"})


# HEALTH CHECK
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    with state_lock:
        return jsonify({
            "status": "healthy",
            "app_name": "SeagrassID",
            "version": "2.0.0",
            "yolo_species": len(YOLO_SPECIES),
            "efficientdet_classes": len(EFFICIENTDET_SPECIES),
            "valid_species": len(VALID_SPECIES),
            "model_loaded": {
                "yolov8": load_yolo() is not None,
                "efficientdet": load_efficientdet() is not None
            },
            "selected_model": app_state["selected_model"]
        })


# ENTRY POINT
if __name__ == "__main__":
    print("=" * 60)
    print("  🌊 SeagrassID — Flask Version (Image Identification)")
    print("  📍 http://localhost:5001")
    print("  🔍 Models:")
    print("     - YOLOv8: 6 seagrass species")
    print("     - EfficientDet-D0: 6 species")
    print("  🌿 Valid Species: 6 seagrass species")
    print("=" * 60)
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5001)