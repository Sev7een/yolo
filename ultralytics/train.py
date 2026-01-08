import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure we import the local repo's `ultralytics` package (not site-packages)
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # .../E:/ultralytics
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO
import ultralytics as _ultralytics


def main():
    print(f"[import check] ultralytics from: {_ultralytics.__file__}")

    # TODO: adjust these paths to your project
    model = YOLO("yolo11n.pt")

    results = model.train(
        data="dataset.yaml",
        epochs=150,
        seed=42,
        batch=16,
        device=0,  # 使用第一个GPU
        workers=6,
        save=True,
        plots=True,
        verbose=True,
        # NWD controls ---------------------------------------------------------------------------------
        nwd=True,  # True 启用 NWD, False 关闭（默认关闭，保持 Ultralytics 原行为）
        nwd_weight=0.5,  # 0=只用CIoU，1=只用NWD，建议 0.3~0.7 之间尝试
        nwd_sigma=1.0,  # exp(-sqrt(d)/sigma) 的 sigma
    )

    print("训练完成！")
    print(f"最佳模型: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
