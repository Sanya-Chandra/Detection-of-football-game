# auto_label.py
"""Generate pseudo-labels for unlabeled images using a pretrained YOLOv8 model.
Usage:
    python scripts/auto_label.py --weights yolov8n.pt --data_dir storage/dataset/football_players/images/train --output_dir storage/dataset/football_players/labels/train
"""
import argparse
import os
import glob
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolov8n.pt", help="Path to YOLOv8 weights")
    parser.add_argument("--data_dir", required=True, help="Directory with images to label")
    parser.add_argument("--output_dir", required=True, help="Directory to save label txt files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = YOLO(args.weights)
    image_paths = glob.glob(os.path.join(args.data_dir, "*.*"))
    for img_path in image_paths:
        # YOLO predict returns a Results object; save labels in YOLO txt format
        results = model.predict(img_path, save=False, conf=0.25)
        # results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        txt_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        with open(txt_path, "w") as f:
            for box, cls, conf in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = box
                f.write(f"{int(cls)} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {conf:.6f}\n")

if __name__ == "__main__":
    main()
