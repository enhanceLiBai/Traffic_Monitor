import cv2
import json
import os
import sys
import shutil
import subprocess
from tkinter import filedialog, Tk

# ==================== 🎯 本地运行配置区 ====================
SCALE = 0.5                               # 画面缩放比例（与检测脚本保持一致）
LOCAL_CONFIG_PATH = "./config.json"       # 本地配置文件路径
LOCAL_SCRIPT_PATH = "traffic_monitor.py"  # 检测脚本路径（请确保与实际脚本名一致）
# ==========================================================

VIDEO_PATH = ""
user_points = []
frame_setup = None


def select_local_video():
    """弹出系统文件选择框让用户挑选视频"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    print("📂 请在弹出的系统窗口中选择你要检测的视频文件...")
    file_path = filedialog.askopenfilename(
        title="选择待检测的视频文件",
        filetypes=[("视频文件", "*.mp4 *.avi *.mkv *.mov"), ("所有文件", "*.*")]
    )
    
    root.update()  # 确保 Tkinter 窗口彻底关闭
    root.destroy()

    if not file_path:
        print("❌ 错误：未选择任何视频，程序安全退出。")
        os._exit(0)
    return file_path


def get_location(event, x, y, flags, param):
    """鼠标回调函数：只负责记录坐标和实时绘制，杜绝耗时操作"""
    global user_points, frame_setup
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(user_points) < 2:
            user_points.append([x, y])
            print(f"📍 已记录端点 {chr(65 + len(user_points)-1)} -> X: {x}, Y: {y}")
            
            # 绘制红点
            cv2.circle(frame_setup, (x, y), 8, (0, 0, 255), -1)

            if len(user_points) == 2:
                # 绘制绿线
                cv2.line(frame_setup, tuple(user_points[0]), tuple(user_points[1]), (0, 255, 0), 2)
                print("\n✅ 划线已完成！")
                print("🔔 [操作提示]：请在图片窗口激活状态下，按【键盘任意键】正式保存并启动检测...")
            
            cv2.imshow("Setup Tool", frame_setup)


def run_local_detection():
    """调用本地检测脚本"""
    print("\n" + "=" * 60)
    print("🔥 正在启动本地交通检测程序...")
    print("=" * 60 + "\n")

    try:
        subprocess.run([sys.executable, LOCAL_SCRIPT_PATH], check=True)
        print("\n✅ 检测程序运行完毕！")

        # 显示输出文件位置
        if os.path.exists("./recognition_results.csv"):
            print("📊 识别结果文件: ./recognition_results.csv")
        if os.path.exists("./debug_capture"):
            print("📷 抓拍图片目录: ./debug_capture/")
        if os.path.exists("./yolo_hy_plate_result.mp4"):
            print("🎬 结果视频文件: ./yolo_hy_plate_result.mp4")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ 检测程序运行失败，错误码: {e.returncode}")
    except FileNotFoundError:
        print(f"\n❌ 找不到检测脚本: {LOCAL_SCRIPT_PATH}")
        print(f"   请确认 {LOCAL_SCRIPT_PATH} 与当前脚本在同一目录下")


def main():
    global frame_setup, VIDEO_PATH

    VIDEO_PATH = select_local_video()

    if not os.path.exists(VIDEO_PATH):
        print(f"❌ 错误：找不到本地视频文件：{VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        print("❌ 错误：无法读取视频帧！")
        return

    orig_height, orig_width, _ = first_frame.shape
    width = int(orig_width * SCALE)
    height = int(orig_height * SCALE)
    frame_setup = cv2.resize(first_frame, (width, height))

    cv2.namedWindow("Setup Tool", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Setup Tool", frame_setup)
    cv2.setMouseCallback("Setup Tool", get_location)

    print("=" * 60)
    print(f"🎬 成功读取本地视频: {os.path.basename(VIDEO_PATH)}")
    print(f"📐 当前推理分辨率设为: {width} × {height}")
    print("👉 请在弹出的窗口中，【依次点击】左端点(A) 和 右端点(B)")
    print("=" * 60)

    # 1. 阻塞等待用户划线完成（期间监控窗口是否被主动关闭）
    is_closed = False
    while len(user_points) < 2:
        if cv2.getWindowProperty("Setup Tool", cv2.WND_PROP_VISIBLE) < 1:
            is_closed = True
            break
        cv2.waitKey(50)

    if is_closed:
        print("\n👋 用户在划线未完成时关闭了窗口，程序退出。")
        cv2.destroyAllWindows()
        os._exit(0)

    # 2. 划线完成，等待用户按键盘任意键确认
    user_confirmed = False
    while True:
        if cv2.getWindowProperty("Setup Tool", cv2.WND_PROP_VISIBLE) < 1:
            # 用户点 X 关闭了窗口，视为取消后续检测
            break
        key = cv2.waitKey(50)
        if key != -1:  # 用户按了键盘
            user_confirmed = True
            break

    cv2.destroyAllWindows()

    # 如果用户没按键盘而是直接点X关闭了窗口，则安全退出，不触发后续逻辑
    if not user_confirmed:
        print("\n👋 用户取消了后续检测，程序安全退出。")
        os._exit(0)

    # 3. 核心IO与配置写入移到主线程（避免卡死回调）
    os.makedirs("./data", exist_ok=True)
    video_filename = os.path.basename(VIDEO_PATH)
    dest_video_path = f"./data/{video_filename}"
    
    print(f"\n📦 正在复制视频到本地目录 (大文件请稍候): {dest_video_path} ...")
    shutil.copy2(VIDEO_PATH, dest_video_path)
    print("✅ 视频复制完成！")

    config_data = {
        "video_path": dest_video_path,
        "scale": SCALE,
        "line_a": user_points[0],
        "line_b": user_points[1],
        "is_configured": True
    }
    with open(LOCAL_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)
    print(f"✅ 配置文件已成功保存到: {LOCAL_CONFIG_PATH}")

    # 4. 启动本地核心检测脚本
    run_local_detection()


if __name__ == "__main__":
    main()