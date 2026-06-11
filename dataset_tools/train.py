from ultralytics import YOLO

def main():
    # 1. 加载你已经下载好的 yolov8m 预训练权重
    model = YOLO("yolov8m.pt")

    # 2. 开始训练
    results = model.train(
        data="/root/autodl-tmp/FlowPlate-Analyzer/data.yaml",  # 你的配置文件路径
        epochs=100,             # 训练轮数，车牌任务 50~100 轮通常就收敛得非常好了
        imgsz=640,              # YOLOv8 默认输入尺寸
        batch=16,               # 批大小。如果换了 3090/4090 等大显存卡，可以改到 32 或 64
        workers=8,              # 数据加载的数据线程数
        device=0,               # 使用第 0 块显卡训练
        optimizer="SGD",        # 优化器，也可以用 'AdamW'
        project="runs/detect",  # 保存训练结果的根目录
        name="flow_plate_m",    # 实验名称，训练完的权重会存在 runs/detect/flow_plate_m/weights/best.pt
        plots=True              # 生成训练曲线图（P-R曲线、损失曲线等）
    )

if __name__ == "__main__":
    main()