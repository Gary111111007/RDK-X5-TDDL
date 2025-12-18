# RDK X5 人体关键点发送脚本（8888端口，IP：192.168.5.6）
import cv2
import mediapipe as mp
import socket
import json
import traceback
COMPUTER_IP = "192.168.5.6"  
PORT = 8888  
CAM_WIDTH = 480
CAM_HEIGHT = 320
VISIBILITY_THRESHOLD = 0.5
# 初始化UDP套接字（设置SO_REUSEADDR，避免端口占用残留）
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  
sock.settimeout(1.0)  

# 初始化MediaPipe姿态检测（轻量化模型，速度优先）
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 打开摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

print("="*50)
print(f"RDK X5 发送端已启动！")
print(f"目标电脑IP: {COMPUTER_IP}:{PORT}")
print(f"摄像头分辨率: {CAM_WIDTH}x{CAM_HEIGHT}")
print("按 'q' 键退出程序")
print("="*50)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("摄像头读取失败，退出...")
            break

        # 镜像翻转画面（操作更直观，和自己的动作一致）
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # MediaPipe处理图像（BGR转RGB，MediaPipe要求RGB格式）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # 提取有效关键点并发送
        if results.pose_landmarks:
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                # 过滤置信度不足的关键点，无效点填充0
                if lm.visibility < VISIBILITY_THRESHOLD:
                    landmarks.append([0.0, 0.0, 0.0])
                    continue
                # 转换为像素坐标，保留2位小数减少数据量
                x = round(lm.x * w, 2)
                y = round(lm.y * h, 2)
                z = round(lm.z * w, 2)  # z轴按x比例缩放，保证尺度一致
                landmarks.append([x, y, z])

            # 发送关键点数据到电脑
            try:
                data = json.dumps(landmarks).encode("utf-8")
                sock.sendto(data, (COMPUTER_IP, PORT))
            except socket.timeout:
                print("警告：数据发送超时（电脑未响应/网络断开）")
            except Exception as e:
                print(f"❌ 发送失败: {str(e)}")

        # 在摄像头画面绘制人体关键点和骨骼连线，方便观察检测效果
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            # 自定义关键点和连线样式（可选）
            mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )
        cv2.imshow("RDK X5 Pose Detection (8888 Port)", frame)

        # 按q键退出程序
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("用户主动退出程序...")
            break

except KeyboardInterrupt:
    print("程序被手动中断...")
except Exception as e:
    print(f"程序异常: {str(e)}")
    traceback.print_exc()  # 打印详细错误信息，方便排查
finally:
    # 释放所有资源，避免内存泄漏
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    sock.close()
    print("✅ 资源已全部释放，程序正常退出")
