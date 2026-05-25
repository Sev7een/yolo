import os
import random
import shutil

# --- 配置区域 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 1. 修改文件夹名称为 dataset1
DATASET_DIR = os.path.join(BASE_DIR, "dataset1")

# 假设你的原始数据目前都在 images/train 和 labels/train 下
# 脚本将从这里把 val 和 test 的数据“挖”走，剩下的就是 train
SOURCE_IMAGES_DIR = os.path.join(DATASET_DIR, "images", "train")
SOURCE_LABELS_DIR = os.path.join(DATASET_DIR, "labels", "train")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 设置随机种子，保证每次划分结果一致
random.seed(42)


def split_dataset():
    # 检查源目录是否存在
    if not os.path.exists(SOURCE_IMAGES_DIR):
        print(f"错误：找不到源图片目录 {SOURCE_IMAGES_DIR}")
        return

    # 获取所有图片文件
    image_files = [f for f in os.listdir(SOURCE_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

    if not image_files:
        print("错误：源目录下没有找到图片文件。")
        return

    # 排序并打乱，确保随机性
    image_files.sort()
    random.shuffle(image_files)

    total = len(image_files)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)
    # 剩下的给测试集

    # 我们只需要把 Val 和 Test 的文件找出来移走，剩下的自然就是 Train
    # 所以这里不需要定义 train_files，它们原地不动即可
    val_files = image_files[train_count : train_count + val_count]
    test_files = image_files[train_count + val_count :]

    print(f"检测到总图片: {total} 张")
    print(f"准备移动 -> 验证集: {len(val_files)} 张, 测试集: {len(test_files)} 张")
    print(f"预计保留 -> 训练集: {total - len(val_files) - len(test_files)} 张")

    # 执行移动操作的函数
    def move_subset(subset_name, files):
        # 创建目标文件夹 (例如 dataset1/images/val)
        target_img_dir = os.path.join(DATASET_DIR, "images", subset_name)
        target_lbl_dir = os.path.join(DATASET_DIR, "labels", subset_name)

        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_lbl_dir, exist_ok=True)

        moved_count = 0
        for img_file in files:
            # 1. 移动图片
            src_img_path = os.path.join(SOURCE_IMAGES_DIR, img_file)
            dst_img_path = os.path.join(target_img_dir, img_file)

            shutil.move(src_img_path, dst_img_path)

            # 2. 移动对应的标签
            # 假设标签和图片同名，只是后缀是 .txt
            label_name = os.path.splitext(img_file)[0] + ".txt"
            src_label_path = os.path.join(SOURCE_LABELS_DIR, label_name)
            dst_label_path = os.path.join(target_lbl_dir, label_name)

            if os.path.exists(src_label_path):
                shutil.move(src_label_path, dst_label_path)
            else:
                print(f"警告: 图片 {img_file} 没有找到对应的标签文件 {label_name}")

            moved_count += 1

        print(f"成功移动 {moved_count} 组数据到 {subset_name}")

    # 开始移动
    if len(val_files) > 0:
        move_subset("val", val_files)

    if len(test_files) > 0:
        move_subset("test", test_files)

    print("\n数据集划分全部完成！")
    print(f"现在 '{SOURCE_IMAGES_DIR}' 中只剩下训练集数据。")


if __name__ == "__main__":
    split_dataset()
