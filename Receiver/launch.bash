#!/bin/bash

# --- ROS 2 环境设置 ---
# 请确保这里的路径是正确的
source install/setup.zsh

# --- 脚本终止处理 ---
# 设置一个trap，当接收到SIGINT信号 (通常是Ctrl+C) 时，会调用'cleanup'函数
trap cleanup SIGINT

# --- 清理函数定义 ---
# 这个函数会杀掉所有由该脚本启动的后台进程
cleanup() {
    echo "正在关闭所有ROS 2节点..."
    # 使用pkill -P $$ 来杀掉所有子进程
    # $$ 是当前脚本的进程ID (PID)
    pkill -P $$
    echo "所有节点已关闭。"
    exit 0
}

# --- 并行启动ROS 2节点 ---
# 使用'&'将每个ros2 run命令放到后台执行

echo "正在启动所有 Handcap Receiver 节点..."

ros2 run handcap_receiver subscriber_node

echo "所有节点已启动。按 Ctrl+C 来关闭所有节点。"

# --- 等待所有后台进程结束 ---
# 'wait'命令会使脚本暂停在这里，直到所有后台任务完成
# 由于ROS节点通常会一直运行，这实际上是在等待用户按Ctrl+C
wait