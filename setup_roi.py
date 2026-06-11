import cv2
import json
import os
import sys
import paramiko  # 用于远程连接服务器
from tkinter import filedialog, Tk  

# ==================== 🖥️ 远程服务器配置区 (已针对 AutoDL 深度优化) ====================
SERVER_IP = "connect.nmb1.seetacloud.com"
SERVER_PORT = 10743
SERVER_USER = "root"
# ⚠️ 优先从环境变量读取密码，避免明文泄露。设置方式：
#    Linux/macOS: export AUTODL_PASS="你的密码"
#    Windows PS:  $env:AUTODL_PASS="你的密码"
SERVER_PASS = os.environ.get("AUTODL_PASS", "")      

SERVER_PROJECT_DIR = "/root/autodl-tmp/FlowPlate-Analyzer"
SERVER_PYTHON_ENV = "python"   
SERVER_SCRIPT_PATH = f"{SERVER_PROJECT_DIR}/traffic_monitor.py"
# ==================================================================================

SCALE = 0.5                        # 画面缩放比例
LOCAL_CONFIG_PATH = "./config.json" # 本地临时保存路径

# 💡 全局变量初始化
VIDEO_PATH = ""
user_points = []
frame_setup = None

def select_local_video():
    """弹出系统文件选择框让用户挑选视频"""
    root = Tk()
    root.withdraw()  # 隐藏 tkinter 的主窗口
    root.attributes('-topmost', True) # 💡 强行让弹窗置顶，防止被掩盖
    
    print("📂 请在弹出的系统窗口中选择你要检测的视频文件...")
    file_path = filedialog.askopenfilename(
        title="选择待检测的视频文件",
        filetypes=[("视频文件", "*.mp4 *.avi *.mkv *.mov"), ("所有文件", "*.*")]
    )
    
    # 彻底销毁 tkinter 实例，释放内存，防止生命周期残留
    root.quit()
    root.destroy()
    
    if not file_path:
        print("❌ 错误：未选择任何视频，程序安全退出。")
        # 💡 使用强力退出，直接干掉当前进程，防止它死循环复活
        os._exit(0) 
    return file_path

def progress_callback(transferred, total):
    """专门用于打印传输进度条的回调函数"""
    percentage = (transferred / total) * 100
    transferred_mb = transferred / (1024 * 1024)
    total_mb = total / (1024 * 1024)
    
    bar_length = 30
    filled_length = int(round(bar_length * transferred / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    sys.stdout.write(f"\r🚀 [SFTP] 正在同步大视频: |{bar}| {percentage:.1f}% ({transferred_mb:.1f}MB / {total_mb:.1f}MB)")
    sys.stdout.flush()
    
    if transferred == total:
        print("\n✅ [SFTP] 视频上传成功！")

def upload_all_and_run(local_video, local_json):
    """自动同步视频、同步配置，并远程触发"""
    try:
        video_filename = os.path.basename(local_video)
        remote_video_path = f"{SERVER_PROJECT_DIR}/data/{video_filename}"
        remote_json_path = f"{SERVER_PROJECT_DIR}/config.json"
        
        print(f"\n⚙️ [💾 路径校准] 正在强行重写本地配置...")
        if os.path.exists(local_json):
            with open(local_json, "r", encoding="utf-8") as f:
                try:
                    config_data = json.load(f)
                except json.JSONDecodeError:
                    config_data = {}
        else:
            config_data = {}

        # 锁定正确的服务器路径
        config_data["video_path"] = remote_video_path
        config_data["is_configured"] = True

        with open(local_json, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
        print("✅ [本地] 配置文件重写成功，已成功锁定新视频路径！")

        print("\n🔄 正在连接远程服务器...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(SERVER_IP, port=SERVER_PORT, username=SERVER_USER, password=SERVER_PASS)
        
        sftp = ssh.open_sftp()
        
        try:
            sftp.mkdir(f"{SERVER_PROJECT_DIR}/data")
        except IOError:
            pass 
            
        print(f"📦 准备处理并上传视频文件: {video_filename}")
        sftp.put(local_video, remote_video_path, callback=progress_callback)
        
        print("📋 [SFTP] 正在同步最新的划线配置文件...")
        sftp.put(local_json, remote_json_path)
        sftp.close()
        
        print("🔥 [SSH] 正在激活服务器 GPU 进行推理识别...")
        cmd = f"source /etc/profile && source ~/.bashrc && cd {SERVER_PROJECT_DIR} && nohup python {SERVER_SCRIPT_PATH} > detection.log 2>&1 &"
        ssh.exec_command(cmd)
        
        print(f"\n🎉 全部搞定！后台检测程序已安全就绪。")
        print(f"📝 提示：你可以去查看 AutoDL 服务器上的 {SERVER_PROJECT_DIR}/detection.log 日志。")
        
        ssh.close()
    except Exception as e:
        print(f"\n❌ 自动化流运行失败，原因: {e}")

def get_location(event, x, y, flags, param):
    global user_points, frame_setup, VIDEO_PATH
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(user_points) < 2:
            user_points.append([x, y])
            print(f"📍 已记录端点 {chr(65 + len(user_points)-1)} -> X: {x}, Y: {y}")
            cv2.circle(frame_setup, (x, y), 8, (0, 0, 255), -1)
            
            if len(user_points) == 2:
                cv2.line(frame_setup, tuple(user_points[0]), tuple(user_points[1]), (0, 255, 0), 3)
                print("\n🎉 本地计数线设置成功！")
                
                cv2.imshow("Setup Tool", frame_setup)
                cv2.waitKey(100) 
                
                config_data = {
                    "video_path": VIDEO_PATH, 
                    "scale": SCALE,
                    "line_a": user_points[0],
                    "line_b": user_points[1],
                    "is_configured": True
                }
                with open(LOCAL_CONFIG_PATH, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, indent=4, ensure_ascii=False)
                
                # 触发一键同步
                upload_all_and_run(VIDEO_PATH, LOCAL_CONFIG_PATH)
                print(f"\n💾 按【任意键】或【右上角关闭】安全退出本地配置工具。")
                
            cv2.imshow("Setup Tool", frame_setup)

def main():
    global frame_setup, VIDEO_PATH

    if not SERVER_PASS:
        print("⚠️ 警告：未设置服务器密码（环境变量 AUTODL_PASS 为空）")
        print("   远程部署功能将不可用。如需使用，请在终端中设置密码后重试：")
        print("   Linux/macOS: export AUTODL_PASS=\"你的密码\"")
        print("   Windows PS:  $env:AUTODL_PASS=\"你的密码\"")
        print("   或在 setup_roi.py 中直接填写 SERVER_PASS = \"你的密码\"\n")

    # 💡 核心改动：移到 main 内部执行，只有运行 main 才会触发弹窗
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

    print("="*60)
    print(f"🎬 成功读取本地视频: {os.path.basename(VIDEO_PATH)}")
    print(f"📐 当前推理分辨率设为: {width} × {height}")
    print("👉 请在弹出的窗口中，【依次点击】左端点(A) 和 右端点(B)")
    print("="*60)

    # 阻塞等待用户划线完成
    while len(user_points) < 2:
        # 如果用户直接点 X 关掉了 OpenCV 窗口，getWindowProperty 会返回 -1
        if cv2.getWindowProperty("Setup Tool", cv2.WND_PROP_VISIBLE) < 1:
            print("\n👋 用户主动关闭了划线窗口，程序退出。")
            os._exit(0)
        cv2.waitKey(100)
        
    # 划线完成后的最终等待退出
    while True:
        # 只要用户按了键盘任意键，或者把 OpenCV 窗口关了，就彻底毁灭程序
        if cv2.waitKey(100) != -1 or cv2.getWindowProperty("Setup Tool", cv2.WND_PROP_VISIBLE) < 1:
            break
            
    cv2.destroyAllWindows()
    print("\n👋 划线配置工具已安全关闭。")
    os._exit(0) # 彻底杀死所有残留线程

if __name__ == "__main__":
    main()