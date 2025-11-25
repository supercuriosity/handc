import time
import getpass # 导入 getpass 库用于安全地输入密码
import sys
import paramiko # 导入 paramiko 库

# --- 配置你的服务器信息 ---
# 建议将密码留空，通过运行时输入来提高安全性
SSH_USER = "root"
SSH_HOST = "192.168.0.116"
SSH_PASSWORD = "orangepi" # 将密码留空或在下面运行时输入
# -------------------------

def get_remote_time(user, host, password):
    """通过SSH(paramiko)连接到远程服务器并获取 time.time()"""
    
    # 构建将在远程服务器上执行的命令
    remote_command = "python -c 'import time; print(time.time())'"
    
    # 创建一个SSH客户端实例
    client = paramiko.SSHClient()
    # 自动添加不在known_hosts文件中的主机密钥（安全性较低，但方便）
    # 在生产环境中，建议使用更安全的方式管理主机密钥
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    print(f"尝试连接到 {user}@{host}...")

    try:
        # 尝试连接服务器
        client.connect(
            hostname=host,
            username=user,
            password=password,
            timeout=10 # 设置连接超时
        )
        
        # 执行命令
        stdin, stdout, stderr = client.exec_command(remote_command)
        
        # 读取命令的标准输出
        output = stdout.read().decode('utf-8').strip()
        # 检查是否有错误输出
        error = stderr.read().decode('utf-8').strip()
        
        if error:
            print(f"错误: 远程命令执行时返回错误: {error}", file=sys.stderr)
            return None
        
        # 将返回的字符串输出转换为浮点数
        remote_timestamp = float(output)
        return remote_timestamp

    except paramiko.AuthenticationException:
        print(f"错误: 身份验证失败！请检查您的用户名和密码。", file=sys.stderr)
        return None
    except paramiko.SSHException as e:
        print(f"错误: 发生SSH错误: {e}", file=sys.stderr)
        return None
    except TimeoutError:
        print(f"错误: 连接到 {host} 超时。", file=sys.stderr)
        return None
    except ValueError:
        print(f"错误: 无法将服务器返回的 '{output}' 转换为浮点数。", file=sys.stderr)
        return None
    except Exception as e:
        print(f"错误: 发生未知错误: {e}", file=sys.stderr)
        return None
    finally:
        # 确保无论成功与否，连接都会被关闭
        if client:
            client.close()


if __name__ == "__main__":
    # --- 安全地获取密码 ---
    # 检查脚本中是否已硬编码密码
    password_to_use = SSH_PASSWORD
    if not password_to_use:
        try:
            # 如果脚本中密码为空，则提示用户输入
            password_to_use = getpass.getpass(f"请输入 {SSH_USER}@{SSH_HOST} 的密码: ")
        except KeyboardInterrupt:
            print("\n操作取消。")
            sys.exit(0)
    else:
        print("警告：直接在脚本中硬编码密码是不安全的做法。")
    
    
    mean_time = []
    for i in range(1000):
        # 1. 获取本地时间戳
        local_time_start = time.time()
        
        # 2. 获取远程时间戳 (传入密码)
        remote_time = get_remote_time(SSH_USER, SSH_HOST, password_to_use)
        
        # 3. 再次获取本地时间戳，以估算网络延迟
        local_time_end = time.time()

        if remote_time is not None:
            print("\n--- 时间戳结果 ---")
            print(f"本地时间 (开始时): {local_time_start:.6f}")
            print(f"服务器时间:          {remote_time:.6f}")
            print(f"本地时间 (结束时): {local_time_end:.6f}")
            
            # 计算差值
            time_diff = remote_time - local_time_start
            mean_time.append(time_diff)
            # round_trip_time = local_time_end - local_time_start
            
            # print("\n--- 分析 ---")
            # print(f"服务器时间与本地时间的表面差值: {time_diff:.6f} 秒")
            # print(f"本次SSH查询的总往返时间 (RTT) 约: {round_trip_time:.6f} 秒")
            # print("\n注意：这个差值包含了网络延迟和命令执行时间。")
            # print("如果两台机器时钟完全同步，差值应约等于单程网络延迟。")
    
    else:
        if mean_time:
            average_diff = sum(mean_time) / len(mean_time)
            print(f"\n在 {len(mean_time)} 次尝试中，服务器时间与本地时间的平均差值约为: {average_diff:.6f} 秒")
        else:
            print("未能成功获取任何远程时间戳。")