import os
import cv2
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np

app = Flask(__name__)

# ================== CONFIG ==================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DENOMINATION_VND = {
    '1k':    1_000,  '2k':    2_000,  '5k':    5_000,
    '10k':  10_000,  '20k':  20_000,  '50k':  50_000,
    '100k': 100_000, '200k': 200_000, '500k': 500_000,
    # Fallback nếu class name là số nguyên
    '1000':    1_000,  '2000':    2_000,  '5000':    5_000,
    '10000':  10_000,  '20000':  20_000,  '50000':  50_000,
    '100000': 100_000, '200000': 200_000, '500000': 500_000,
}

ALPHA = 0.4   # trọng số YOLO trong fusion

# Màu cho từng mệnh giá (BGR cho OpenCV)
COLORS_BGR = [
    (0, 200, 100),   (0, 150, 255),  (255, 100, 0),
    (200, 0, 200),   (0, 200, 200),  (255, 50, 50),
    (50, 200, 255),  (255, 180, 0),  (100, 255, 50),
]

# ================== LOAD YOLO ==================
print("📥 Loading YOLO...")
yolo_model = YOLO("best.pt")
YOLO_NAMES = yolo_model.names  # {0: '100k', ...}
print(f"✅ YOLO loaded — classes: {YOLO_NAMES}")

# ================== LOAD EFFICIENTNET ==================
print("📥 Loading EfficientNetB0...")
checkpoint = torch.load("efficientnet_b0_cbam_vnd9_best.pth", map_location='cpu')

# Lấy class mapping từ checkpoint
if "idx_to_class" in checkpoint:
    idx_to_class = checkpoint["idx_to_class"]
    # idx_to_class có thể là {0: 'x', 1: 'y'} hoặc {'0': 'x', '1': 'y'}
    if all(isinstance(k, str) for k in idx_to_class.keys()):
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    EFF_CLASSES = [idx_to_class[i] for i in range(len(idx_to_class))]
elif "class_to_idx" in checkpoint:
    class_to_idx = checkpoint["class_to_idx"]
    EFF_CLASSES = [k for k, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
else:
    EFF_CLASSES = ['1000','2000','5000','10000','20000','50000','100000','200000','500000']

NUM_CLASSES = len(EFF_CLASSES)
print(f"✅ EfficientNet classes ({NUM_CLASSES}): {EFF_CLASSES}")

# ── CBAM MODULE — copy y chang từ notebook train ──
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(
            self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        )

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class EfficientNetB0_CBAM(nn.Module):
    def __init__(self, num_classes=9, dropout=0.3):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        self.features   = base.features
        self.cbam       = CBAM(1280)
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

# Build model — đúng kiến trúc train
eff_model = EfficientNetB0_CBAM(num_classes=NUM_CLASSES, dropout=0.3)
eff_model.load_state_dict(checkpoint["model_state_dict"])
eff_model.eval()
print("✅ EfficientNetB0 + CBAM loaded!")

# Transform cho crop
crop_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ================== CLASSIFY CROP ==================
def classify_crop(crop_bgr):
    """Classify một crop ảnh BGR → (class_name, confidence, all_probs)"""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img  = Image.fromarray(crop_rgb)
    tensor   = crop_transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        logits = eff_model(tensor)
        probs  = F.softmax(logits, dim=1)[0].numpy()

    pred_idx  = int(np.argmax(probs))
    pred_name = EFF_CLASSES[pred_idx]
    pred_conf = float(probs[pred_idx])

    return pred_name, pred_conf, probs

# ================== PIPELINE ==================
def process_image(img_bgr):
    """
    Pipeline đầy đủ: YOLO detect → EfficientNet classify → Fusion.
    Trả về: (ảnh đã vẽ, list detections, tổng VND)
    """
    H, W = img_bgr.shape[:2]

    # STEP 1: YOLO detect
    yolo_res = yolo_model(img_bgr, conf=0.3, iou=0.45, verbose=False)[0]
    boxes    = yolo_res.boxes

    if len(boxes) == 0:
        return img_bgr, [], 0

    result_img  = img_bgr.copy()
    detections  = []
    total_value = 0

    for idx_box, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        yolo_cls_id   = int(box.cls[0])
        yolo_conf     = float(box.conf[0])
        yolo_cls_name = YOLO_NAMES[yolo_cls_id]

        # STEP 2: EfficientNet trên crop
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        eff_cls_name, eff_conf, eff_probs = classify_crop(crop)

        # STEP 3: Weighted fusion
        fusion_scores = (1 - ALPHA) * eff_probs.copy()
        if yolo_cls_name in EFF_CLASSES:
            fusion_scores[EFF_CLASSES.index(yolo_cls_name)] += ALPHA * yolo_conf

        final_idx      = int(np.argmax(fusion_scores))
        final_cls_name = EFF_CLASSES[final_idx]
        final_conf     = float(fusion_scores[final_idx])
        agree          = (yolo_cls_name == eff_cls_name)

        vnd_value = DENOMINATION_VND.get(final_cls_name, 0)
        total_value += vnd_value

        # Vẽ lên ảnh
        color = COLORS_BGR[idx_box % len(COLORS_BGR)]
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 3)

        label    = f"{final_cls_name}  {final_conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(result_img, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
        cv2.putText(result_img, label, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        detections.append({
            'index':      idx_box + 1,
            'bbox':       [x1, y1, x2, y2],
            'yolo_class': yolo_cls_name,
            'yolo_conf':  round(yolo_conf * 100, 1),
            'eff_class':  eff_cls_name,
            'eff_conf':   round(eff_conf * 100, 1),
            'final_class': final_cls_name,
            'final_conf':  round(final_conf * 100, 1),
            'value':      vnd_value,
            'agree':      agree,
        })

    # Vẽ tổng tiền ở dưới
    footer = f"Tong: {total_value:,} VND  |  {len(detections)} to tien"
    cv2.rectangle(result_img, (0, H - 44), (W, H), (20, 20, 40), -1)
    cv2.putText(result_img, footer, (10, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

    return result_img, detections, total_value

# ================== ROUTES ==================
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Không có file"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "File trống"}), 400

    # Lưu ảnh gốc
    filename  = file.filename
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    # Đọc ảnh
    img_bgr = cv2.imread(save_path)
    if img_bgr is None:
        return jsonify({"error": "Không đọc được ảnh"}), 400

    # Chạy pipeline
    result_img, detections, total_value = process_image(img_bgr)

    # Lưu ảnh kết quả
    result_filename = "result_" + filename
    result_path     = os.path.join(UPLOAD_FOLDER, result_filename)
    cv2.imwrite(result_path, result_img)

    return jsonify({
        "result_img":  result_path,
        "detections":  detections,
        "total_value": total_value,
        "total_bills": len(detections),
        "agree_count": sum(1 for d in detections if d["agree"]),
    })

# ================== RUN ==================
if __name__ == "__main__":
    app.run(debug=True, port=5000)