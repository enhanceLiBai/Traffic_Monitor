import os
import cv2


def convert_coordinate(img_size, bbox):
    """将 CCPD 的绝对坐标 bbox 转换为 YOLO 的归一化中心点坐标"""
    dw = 1.0 / img_size[0]
    dh = 1.0 / img_size[1]

    # bbox 格式: [xmin, ymin, xmax, ymax]
    x_center = (bbox[0] + bbox[2]) / 2.0
    y_center = (bbox[1] + bbox[3]) / 2.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # 归一化
    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh
    return x, y, w, h


def process_dataset(data_dir):
    """遍历指定文件夹，解析文件名并生成 YOLO 标签"""
    print(f"正在处理文件夹: {data_dir}...")
    if not os.path.exists(data_dir):
        print(f"路径不存在: {data_dir}，跳过。")
        return

    file_list = os.listdir(data_dir)
    img_count = 0

    for file_name in file_list:
        if not file_name.endswith(".jpg"):
            continue

        img_count += 1
        img_path = os.path.join(data_dir, file_name)

        # 1. 读取图片获取宽高（CCPD 2019/2020 绝大多数是 720x1160，但用 cv2 读取最稳妥）
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue
        h, w, _ = img.shape

        try:
            # 2. 分割文件名获取边界框段
            # 示例: 25-91_95-131&428_599&571-... -> 拿取索引为 2 的段: '131&428_599&571'
            annotations = file_name.split("-")
            bbox_str = annotations[2]

            # 3. 解析出左上和右下点
            left_top, right_bottom = bbox_str.split("_")
            xmin, ymin = map(int, left_top.split("&"))
            xmax, ymax = map(int, right_bottom.split("&"))

            # 4. 转换为 YOLO 格式
            x, y, bbox_w, bbox_h = convert_coordinate((w, h), [xmin, ymin, xmax, ymax])

            # 5. 写入对应的 .txt 文件（车牌检测属于单类任务，类别 ID 设为 0）
            txt_name = os.path.splitext(file_name)[0] + ".txt"
            txt_path = os.path.join(data_dir, txt_name)

            with open(txt_path, "w") as f:
                f.write(f"0 {x:.6f} {y:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")

        except Exception as e:
            print(f"解析文件 {file_name} 失败，错误信息: {e}")

    print(f"✨ 完成！共处理了 {img_count} 张图片，并生成了对应的标签文件。")


if __name__ == "__main__":
    # 根据你上一跑完分组后的真实基础路径进行调整
    base_path = "/root/autodl-tmp/FlowPlate-Analyzer/data"

    # 分别处理三个子集
    process_dataset(os.path.join(base_path, "train"))
    process_dataset(os.path.join(base_path, "val"))
    process_dataset(os.path.join(base_path, "test"))
    print("🎉 所有数据处理完毕！")