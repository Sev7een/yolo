import os

from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == "__main__":
    # 1) 在这里手动填写你要测试的 best.pt 路径（建议用原始字符串 r"..."，避免 \t \n 等转义）
    weights_path = r"E:\ultralytics\ultralytics\runs\detect\train23\weights\best.pt"

    # 2) 数据集配置文件
    data_yaml = "dataset.yaml"

    # 3) 测试集评估：这里明确用 split='test'
    # 只有当 dataset.yaml 里定义了 test: ... 并且对应目录存在时才会跑测试集。
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml, split="test")

    print(f"结果保存在: {metrics.save_dir}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
