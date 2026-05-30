import argparse
import os
import shutil
import torch
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Football Players Dataset")
    parser.add_argument("--data", default="storage/dataset/football_players/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--weights", default="yolov8n.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    # 1. Device selection
    if torch.cuda.is_available():
        device = 0
        print(f"CUDA GPU detected. Training on: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("No CUDA GPU detected. Falling back to CPU training.")

    # 2. Load model
    print(f"Loading base model: {args.weights}")
    model = YOLO(args.weights)

    # 3. Start training
    print(f"Starting training on data: {args.data} for {args.epochs} epochs...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project="runs/detect",
        name="football_players",
        exist_ok=True,
    )

    # 4. Copy best weights to models/weights/
    dest_weights = Path("models/weights/football_players_best.pt")
    
    # Dynamically find the best.pt weights file under runs/
    best_weights_src = None
    for path in Path("runs").rglob("best.pt"):
        # Make sure we pick a file inside the correct subfolder
        if "football_players" in str(path):
            best_weights_src = path
            break

    if best_weights_src and best_weights_src.exists():
        dest_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights_src, dest_weights)
        print(f"\nTraining completed successfully!")
        print(f"Best weights copied to: {dest_weights.resolve()}")
    else:
        print(f"\nWarning: Could not find trained weights in runs/ directory.")

if __name__ == "__main__":
    main()
