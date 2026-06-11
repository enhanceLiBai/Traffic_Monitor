import cv2
import numpy as np
from ultralytics import YOLO
import os
import shutil
import csv
import json
import hyperlpr3 as lpr 
from PIL import Image, ImageDraw, ImageFont

# 如果在完全无显示器的 Linux 服务器上运行，可以取消下面这行的注释
# os.environ["QT_QPA_PLATFORM"] = "offscreen"

# ==================== 📁 读取桥接配置文件 ====================
CONFIG_PATH = "./config.json"
if not os.path.exists(CONFIG_PATH):
    print(f"❌ 错误：找不到配置文件 {CONFIG_PATH}！请先运行 setup_roi.py 进行划线配置。")
    exit()

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

video_path = config["video_path"]
SCALE = config["scale"]
LINE_A = tuple(config["line_a"])
LINE_B = tuple(config["line_b"])

print("="*50)
print(f"📋 成功加载配置：\n🎬 视频: {video_path}\n📐 缩放比: {SCALE}\n📍 计数线A端点: {LINE_A}\n📍 计数线B端点: {LINE_B}")
print("="*50)

# ==================== 🤖 模型与字体初始化区域 ====================
model = YOLO("yolov8m.pt")


# 车牌检测模型（支持 .pt 和 .onnx 两种格式）
# 使用 ONNX 格式可获得更快的推理速度，详见部署文档第 4.6 节 ONNX 导出指南
my_plate_detector = YOLO("./best.onnx") 
hy_recognizer = lpr.LicensePlateCatcher()

# 中文字体路径自动探测（跨平台兼容）
_FONT_CANDIDATES = [
    # Linux: Wqy-Zenhei（文档推荐，apt-get install fonts-wqy-zenhei）
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    # Linux: Noto CJK（部分发行版默认）
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    # Windows 系统自带中文字体
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
]
FONT_PATH = None
for _fp in _FONT_CANDIDATES:
    if os.path.exists(_fp):
        FONT_PATH = _fp
        print(f"🔤 检测到中文字体: {_fp}")
        break
if FONT_PATH is None:
    print("⚠️ 未找到中文字体！车牌中文渲染将降级为英文显示（Linux 请执行: apt-get install fonts-wqy-zenhei）")

# 📊 识别率统计指标
total_captured_vehicles = 0   # 总抓拍车位数
successfully_recognized = 0   # 成功识别车牌数

# 各颜色车牌计数器字典
color_stats = {
    "蓝色": 0,
    "绿色": 0,
    "黄色": 0,
    "白色": 0,
    "黑色": 0,
    "未知": 0
}

# 自动清理旧的 Debug 文件夹
DEBUG_DIR = "./debug_capture"
if os.path.exists(DEBUG_DIR):
    shutil.rmtree(DEBUG_DIR)
os.makedirs(DEBUG_DIR, exist_ok=True)

csv_path = "./recognition_results.csv"
# 初始化 CSV 文件并写入表头
with open(csv_path, mode='w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["车辆ID", "系统识别车牌", "车牌颜色", "真实车牌(手动填写)"])

# ==================== 🧮 几何、文字及颜色辅助函数 ====================

# ✨ 优化点 2：极其关键！重构中文水印渲染函数（由全局图转换改为局部 ROI 转换）
def draw_chinese_text(img, text, position, font_path, font_size=20, color=(0, 255, 0)):
    if font_path is None or not os.path.exists(font_path):
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return img
    
    x, y = position
    # 估算文本所占像素宽高，防止越界
    text_w = font_size * len(text)
    text_h = int(font_size * 1.5)
    
    h, w = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + text_w), min(h, y + text_h)
    
    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return img
        
    # 🧠 只切出文字区域的小局部块进行 Pillow 转换，消耗的 CPU 算力微乎其微！
    roi = img[y1:y2, x1:x2]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(roi_rgb)
    
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)
    rgb_color = (color[2], color[1], color[0])
    
    # 在局部块的左上角写入（需重设相对坐标）
    draw.text((x - x1, y - y1), text, font=font, fill=rgb_color)
    
    # 转换为 numpy 并刷回原图
    roi_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img[y1:y2, x1:x2] = roi_bgr
    return img

def is_near_line(cx, cy, line_a, line_b, max_dist=15):
    x1, y1 = line_a
    x2, y2 = line_b
    num = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1)
    den = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
    if den == 0: return False
    return (num / den) < max_dist

def estimate_color_by_hsv(plate_img):
    if plate_img is None or plate_img.size == 0:
        return "未知"
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    color_ranges = {
        "蓝色": [(100, 50, 50), (140, 255, 255)],
        "黄色": [(11, 60, 60), (34, 255, 255)],
        "绿色": [(35, 40, 40), (85, 255, 255)],
        "白色": [(0, 0, 180), (180, 30, 255)]
    }
    color_counts = {}
    for color_name, (lower, upper) in color_ranges.items():
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_np, upper_np)
        color_counts[color_name] = cv2.countNonZero(mask)
    max_color = max(color_counts, key=color_counts.get)
    if color_counts[max_color] < (plate_img.shape[0] * plate_img.shape[1] * 0.05):
        return "未知"
    return max_color

def parse_plate_color(hy_res_item, plate_crop_img):
    color_str = "未知"
    if len(hy_res_item) >= 4:
        raw_color = hy_res_item[3]
        if isinstance(raw_color, str):
            color_str = raw_color
        elif isinstance(raw_color, int):
            color_map = {0: "蓝色", 1: "黄色", 2: "绿色", 3: "白色", 4: "黑色"}
            color_str = color_map.get(raw_color, "未知")
            
    plate_text = hy_res_item[0]
    if color_str == "未知" and len(plate_text) >= 7:
        if len(plate_text) == 8:
            color_str = "绿色"  
        elif plate_text.startswith("使") or plate_text.endswith("领"):
            color_str = "黑色"
        elif plate_text.endswith("学") or plate_text.endswith("警"):
            color_str = "黄色" if plate_text.endswith("学") else "白色"
            
    if color_str == "未知" or color_str == "":
        color_str = estimate_color_by_hsv(plate_crop_img)
    return color_str

# ==================== 🎥 主循环检测模式 ====================
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

width = int(orig_width * SCALE)
height = int(orig_height * SCALE)

output_path = "yolo_hy_plate_result.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

counted_vehicles_db = {}  
FAILED_FLAG = "车牌污染严重"
ALLOWED_CLASSES = [2, 5, 7]

print("🚀 开始运行核心检测与追踪...")

while True:
    ret, frame_orig = cap.read()
    if not ret: break
        
    frame = cv2.resize(frame_orig, (width, height))
    
    # ✨ 优化点 3：强制追踪模型显式声明走显卡推理 device=0
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, device=0)
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)
        
        for box, track_id, cls in zip(boxes, ids, clss):
            if cls not in ALLOWED_CLASSES: continue
                
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            
            # 实时画面文字绘制逻辑
            if track_id in counted_vehicles_db:
                v_info = counted_vehicles_db[track_id]
                plate_text = v_info["plate"]
                plate_color = v_info["color"]
                display_str = f"{plate_text} ({plate_color})" if plate_text != FAILED_FLAG else FAILED_FLAG
                # ✨ 这里调用了重构后的局部渲染，CPU 负担大幅下降
                frame = draw_chinese_text(frame, display_str, (int(x1), int(y1) - 30), FONT_PATH, font_size=20, color=(0, 255, 0))
            else:
                cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # 线段触发判定
            if is_near_line(cx, cy, LINE_A, LINE_B, max_dist=15):
                if (track_id not in counted_vehicles_db) or (counted_vehicles_db[track_id]["plate"] == FAILED_FLAG):
                    
                    # 映射回原图坐标
                    ox1, oy1 = max(0, int(x1 / SCALE)), max(0, int(y1 / SCALE))
                    ox2, oy2 = min(orig_width, int(x2 / SCALE)), min(orig_height, int(y2 / SCALE))
                    car_crop_4k = frame_orig[oy1:oy2, ox1:ox2]
                    
                    if car_crop_4k.size > 0:
                        final_plate_str = FAILED_FLAG
                        final_color_str = "未知"
                        is_success = False
                        plate_crop_4k = None
                        
                        # ✨ 优化点 4：强制你自己的车牌定位模型跑在显卡上 device=0
                        plate_results = my_plate_detector(car_crop_4k, verbose=False, device=0)
                        if len(plate_results[0].boxes) > 0:
                            px1, py1, px2, py2 = plate_results[0].boxes.xyxy[0].cpu().numpy()
                            ph, pw = car_crop_4k.shape[:2]
                            
                            PADDING = 12
                            kpx1, kpy1 = max(0, int(px1) - PADDING), max(0, int(py1) - PADDING)
                            kpx2, kpy2 = min(pw, int(px2) + PADDING), min(ph, int(py2) + PADDING)
                            
                            plate_crop_4k = car_crop_4k[kpy1:kpy2, kpx1:kpx2]
                            if plate_crop_4k.size > 0:
                                hy_res = hy_recognizer(plate_crop_4k)
                                if len(hy_res) > 0:
                                    final_plate_str = hy_res[0][0]
                                    final_color_str = parse_plate_color(hy_res[0], plate_crop_4k)
                                    is_success = True
                                else:
                                    final_color_str = estimate_color_by_hsv(plate_crop_4k)

                        if track_id not in counted_vehicles_db:
                            total_captured_vehicles += 1
                        
                        counted_vehicles_db[track_id] = {
                            "plate": final_plate_str,
                            "color": final_color_str,
                            "has_reported": False  
                        }
                        
                        # 保存 Debug 图片
                        if is_success:
                            cv2.imwrite(f"{DEBUG_DIR}/ID_{track_id}_car_SUCCESS.jpg", car_crop_4k)
                            if plate_crop_4k is not None:
                                cv2.imwrite(f"{DEBUG_DIR}/ID_{track_id}_plate_SUCCESS_{final_plate_str}_{final_color_str}.jpg", plate_crop_4k)
                        else:
                            cv2.imwrite(f"{DEBUG_DIR}/ID_{track_id}_car_FAILED.jpg", car_crop_4k)
                            if plate_crop_4k is not None:
                                cv2.imwrite(f"{DEBUG_DIR}/ID_{track_id}_plate_FAILED.jpg", plate_crop_4k)

            # 核心漏洞修复逻辑 —— 当车辆离开过线区域或状态稳定后，单独写入 CSV 和统计指标
            if track_id in counted_vehicles_db and not counted_vehicles_db[track_id]["has_reported"]:
                v_info = counted_vehicles_db[track_id]
                if not is_near_line(cx, cy, LINE_A, LINE_B, max_dist=15) or v_info["plate"] != FAILED_FLAG:
                    
                    if v_info["plate"] != FAILED_FLAG:
                        successfully_recognized += 1
                        if v_info["color"] in color_stats:
                            color_stats[v_info["color"]] += 1
                        else:
                            color_stats["未知"] += 1
                    
                    print(f"📷 业务结算 -> 车辆ID: {track_id} | 车牌号码: {v_info['plate']} | 车牌颜色: {v_info['color']}")
                    with open(csv_path, mode='a', encoding='utf-8-sig', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([track_id, v_info["plate"], v_info["color"], ""])
                    
                    counted_vehicles_db[track_id]["has_reported"] = True

    # 绘制ROI基准线
    cv2.line(frame, LINE_A, LINE_B, (0, 255, 0), 4)
    cv2.circle(frame, LINE_A, 8, (0, 0, 255), -1)
    cv2.circle(frame, LINE_B, 8, (0, 0, 255), -1)
    cv2.putText(frame, f"Total Cars: {total_captured_vehicles}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    out.write(frame)

cap.release()
out.release()

# ==================== 📊 计算最终统计指标并追加写入 CSV ====================
success_rate_str = "0.00%"
if total_captured_vehicles > 0:
    success_rate = (successfully_recognized / total_captured_vehicles) * 100
    success_rate_str = f"{success_rate:.2f}%"

with open(csv_path, mode='a', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([]) 
    writer.writerow(["=== 📊 运行结果统计报告 ===", "", "", ""])
    writer.writerow(["触发过线抓拍的车辆总数", f"{total_captured_vehicles} 辆", "", ""])
    writer.writerow(["成功识别到有效车牌数", f"{successfully_recognized} 辆", "", ""])
    writer.writerow(["📈 整体车牌有效识别成功率", success_rate_str, "", ""])
    writer.writerow([])
    writer.writerow(["--- 🎨 车牌颜色多维分布情况 ---", "", "", ""])
    for color_name, color_count in color_stats.items():
        writer.writerow([f"{color_name}车牌总计", f"{color_count} 辆", "", ""])

print("\n" + "="*20 + " 📊 运行结果统计报告 " + "="*20)
print(f"🚗 触发过线抓拍的车辆总数: {total_captured_vehicles} 辆")
print(f"✅ 成功识别到有效车牌数: {successfully_recognized} 辆")
print(f"📈 整体车牌有效识别成功率: {success_rate_str}")
print("\n🎨 车牌颜色详细分类数量：")
for color_name, color_count in color_stats.items():
    print(f"    - {color_name}车牌: {color_count} 辆")
print("="*58)