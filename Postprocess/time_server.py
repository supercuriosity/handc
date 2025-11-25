import socket
import datetime

# 创建一个 socket 对象
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 获取本地主机名和设置端口
host = '0.0.0.0'  # 监听所有网络接口
port = 12345
server_socket.bind((host, port))

# 设置最大连接数
server_socket.listen(5)
print(f"时间服务器正在监听端口 {port}...")

while True:
    # 建立客户端连接
    client_socket, addr = server_socket.accept()
    print(f"接收到来自 {addr} 的连接")

    # 获取当前精确时间并编码为UTF-8发送
    current_time = datetime.datetime.now().isoformat()
    client_socket.send(current_time.encode('utf-8'))
    
    # 关闭连接
    client_socket.close()