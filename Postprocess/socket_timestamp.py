import paramiko
import socket
import datetime
import time
import statistics

# --- 1. SSH 和 远程服务器配置 ---
PI_IP = "192.168.0.116"
PI_USER = "root"
# !! 安全提示: 直接在代码中写密码是不推荐的最佳实践。
# !! 更安全的方式是使用SSH密钥（见下文说明）。
PI_PASSWORD = "orangepi" # <--- 在这里替换成你树莓派的密码

# 远程服务器脚本的路径
REMOTE_SCRIPT_PATH = "/root/workspace/handcap/Postprocess/time_server.py" 
# 请确保您的 time_server.py 文件确实在这个路径下

# --- 2. 客户端配置 (和您之前的脚本一样) ---
PORT = 12345
NUM_ITERATIONS = 100

# ==============================================================================
#  主程序
# ==============================================================================

# 创建一个SSH客户端实例
ssh_client = paramiko.SSHClient()
# 自动添加主机密钥，简化连接
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    # --- 第 A 部分: 自动SSH连接并启动远程服务器 ---
    print(f"[*] 正在通过SSH连接到 {PI_USER}@{PI_IP}...")
    # 使用密码进行连接
    ssh_client.connect(hostname=PI_IP, username=PI_USER, password=PI_PASSWORD)
    print("[+] SSH连接成功！")

    # 使用 nohup 和 & 在后台启动服务器，这样即使SSH会话关闭，服务器也能继续运行
    command = f"nohup python3 {REMOTE_SCRIPT_PATH} > /dev/null 2>&1 &"
    print(f"[*] 正在树莓派上执行命令: {command}")
    ssh_client.exec_command(command)
    
    # 等待一小会儿，确保服务器有足够的时间启动并开始监听
    print("[*] 等待2秒钟，确保远程服务器已启动...")
    time.sleep(2)

    # --- 第 B 部分: 运行本地客户端进行测量 (这是您之前的代码) ---
    time_differences_seconds = []
    successful_measurements = 0
    failed_attempts = 0

    print(f"\n[*] 开始进行 {NUM_ITERATIONS} 次时钟差测量...")
    for i in range(NUM_ITERATIONS):
        try:
            pc_time_before = datetime.datetime.now()
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect((PI_IP, PORT))
            pi_time_str = s.recv(1024).decode('utf-8')
            pc_time_after = datetime.datetime.now()
            s.close()

            pi_time = datetime.datetime.fromisoformat(pi_time_str)
            pc_time_avg = pc_time_before + (pc_time_after - pc_time_before) / 2
            time_difference = pi_time - pc_time_avg
            
            time_differences_seconds.append(time_difference.total_seconds())
            successful_measurements += 1
        except Exception as e:
            # 简化错误处理
            failed_attempts += 1
        # 打印一个简单的进度
        print(f"\r    测量进度: {i+1}/{NUM_ITERATIONS}", end="")
        time.sleep(0.05) # 稍微缩短等待时间

    print("\n\n" + "="*30)
    print("测量完成，开始统计结果...")
    print("="*30)
    
    if successful_measurements > 0:
        average_diff = statistics.mean(time_differences_seconds)
        min_diff = min(time_differences_seconds)
        max_diff = max(time_differences_seconds)
        std_dev = statistics.stdev(time_differences_seconds) if successful_measurements > 1 else 0.0

        print(f"成功测量次数: {successful_measurements}")
        print(f"失败尝试次数: {failed_attempts}")
        print("-" * 30)
        print(f"平均时钟差: {average_diff:+.9f} 秒")
        print(f"最小时钟差: {min_diff:+.9f} 秒")
        print(f"最大时钟差: {max_diff:+.9f} 秒")
        print(f"标准差 (稳定性): {std_dev:.9f} 秒")
    else:
        print("所有测量均失败！")


finally:
    # --- 第 C 部分: 清理工作，关闭远程服务器和SSH连接 ---
    print("\n[*] 测试完成，正在关闭远程服务器...")
    # 找到服务器进程的PID并杀掉它
    # pidof 命令会返回运行指定程序的进程ID
    stdin, stdout, stderr = ssh_client.exec_command(f"pidof python3")
    pids = stdout.read().decode().strip()
    if pids:
        print(f"[*] 找到服务器进程PID: {pids}。正在终止...")
        ssh_client.exec_command(f"kill {pids}")
    else:
        print("[!] 未能在树莓派上找到服务器进程。可能已手动关闭。")

    print("[*] 关闭SSH连接。")
    ssh_client.close()