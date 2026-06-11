import cv2
import json
import os
import sys
import shutil
import subprocess
from tkinter import filedialog, Tk

# ==================== 🎯 本地运行配置区 ====================
SCALE = 0.5                              # 画面缩放比例（与检测脚本保持一致）
LOCAL_CONFIG_PATH = "./config.json"       # 本地配置文件路径
LOCAL_SCRIPT_PATH = "traffic_monitor.py"  # 检测脚本路径
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

    root.quit()
    root.destroy()

    if not file_path:
        print("❌ 错误：未选择任何视频，程序安全退出。")
        os._exit(0)
    return file_path


def get_location(event, x, y, flags, param):
    global user_points, frame_setup, VIDEO_PATH
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(user_points) < 2:
            user_points.append([x, y])
            print(f"📍 已记录端点 {chr(65 + len(user_points)-1)} -> X: {x}, Y: {y}")
            cv2.circle(frame_setup, (x, y), 8, (0, 0, 255), -1)

            if len(user_points) == 2:
                cv2.line(frame_setup, tuple(user_points[0]), tuple(user_points[1]), (0, 255, 0), 3)
                cv2.imshow("Setup Tool", frame_setup)
                cv2.waitKey(100)

                # 1. 复制视频到 data 目录
                os.makedirs("./data", exist_ok=True)
                video_filename = os.path.basename(VIDEO_PATH)
                dest_video_path = f"./data/{video_filename}"
                print(f"\n📦 正在复制视频到本地: {dest_video_path} ...")
                shutil.copy2(VIDEO_PATH, dest_video_path)
                print("✅ 视频复制完成！")

                # 2. 写入 config.json
                config_data = {
                    "video_path": dest_video_path,
                    "scale": SCALE,
                    "line_a": user_points[0],
                    "line_b": user_points[1],
                    "is_configured": True
                }
                with open(LOCAL_CONFIG_PATH, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, indent=4, ensure_ascii=False)
                print(f"✅ 配置文件已保存: {LOCAL_CONFIG_PATH}")

                # 3. 关闭划线窗口
                print("\n🔔 按【任意键】关闭划线窗口并启动检测程序...")

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
        print("   请确认 setup_roi_local.py 与 traffic_plate_counter.py 在同一目录下")


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

    # 阻塞等待用户划线完成
    while len(user_points) < 2:
        if cv2.getWindowProperty("Setup Tool", cv2.WND_PROP_VISIBLE) < 1:
            print("\n👋 用户主动关闭了划线窗口，程序退出。")
            os._exit(0)
        cv2.waitKey(100)

    # 划线完成，等待用户按键启动检测
    while True:
        key = cv2.waitKey(100)
        if key != -1 or cv2.getWindowProperty("Setup Tool", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

    # 启动本地检测
    run_local_detection()


if __name__ == "__main__":
    main()
