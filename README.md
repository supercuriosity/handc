# HandCap

HandCap 是一个用于手持双臂多传感器接口的数据采集系统。该项目旨在提供高效、同步的多模态数据录制功能，支持触觉、视觉、力觉、听觉以及位姿等多种传感器数据的采集、处理与可视化。

## 🌟 功能特性

*   **多模态传感器同步采集**: 支持同时采集触觉相机、腕部相机、力传感器、角度传感器和音频数据。
*   **高性能录制**: 基于多进程/多线程架构，确保高帧率数据写入，支持视频 (`.mp4`) 和元数据 (`.json`, `.npz`) 的高效存储。
*   **先进算法支持**: 数据格式兼容 **LeRobot** (支持 Diffusion Policy, ACT 等) 和 **UMI** (支持 Pi0 等) 训练框架。
*   **灵活配置**: 通过 `config.py` 轻松管理设备路径、ID 和参数。
*   **完整的数据处理流水线**: 提供从原始数据到 LeRobot 或 UMI 训练格式的转换工具。
*   **可视化工具**: 内置多种可视化脚本，用于检查传感器状态和数据质量。
*   **ROS2 集成**: 提供 `data_recorder.py` 用于在 ROS2 环境下进行数据同步录制。

## 📂 项目结构

```
HandCap/
├── handcap.py                  # 主采集程序，直接操作硬件接口
├── config.py                   # 硬件设备配置文件
├── data_recorder.py            # 基于 ROS2 的数据录制节点
├── Sensor/                     # 传感器驱动模块
│   ├── angle.py                # 角度传感器驱动
│   ├── camera_process.py       # 相机驱动 (Tactile, Wrist)
│   ├── force.py                # 力传感器驱动
│   └── ...
├── Postprocess/                # 数据后处理工具
│   ├── _0_combine_and_transfer_data_into_lerobot.py  # 转换为 LeRobot 格式 (Diffusion Policy)
│   ├── _02_combine_and_transfer_data_into_umi.py     # 转换为 UMI 格式 (Pi0)
│   └── ...
├── eval_real/                  # 真机评估脚本
│   ├── eval_real_flexiv_pi0.py # Pi0 模型真机评估
│   └── ...
├── Prepocess/                  # 数据预处理工具 (校准、延迟计算等)
├── common/                     # 通用工具类 (数据模型、时间、可视化)
├── test_sensor/                # 传感器测试与调试脚本
└── example_data/               # 示例数据目录
```

## 🛠️ 硬件支持

项目主要支持以下硬件设备：

*   **Tactile Camera**: USB 接口触觉相机 (支持左右手)。
*   **Wrist Camera**: USB 接口腕部相机。
*   **Force Sensor**: I2C 接口力传感器 (支持左右手)。
*   **Angle Sensor**: I2C 接口角度传感器 (如 AS5600)。
*   **HTC Vive Tracker**: 用于采集手部或设备的 6-DoF 位姿。
*   **Audio**: 系统音频输入设备。

## 🚀 使用指南

### 1. 配置硬件

在运行采集程序之前，请根据实际硬件连接修改 `config.py` 文件：

```python
# config.py 示例
TACTILE_CAMERA = {
    "left": "/dev/video2",
    "right": "/dev/video0"
}

FORCE_SENSOR = {
    "left": {"smbus_id": 3, "i2c_address": 0x48},
    "right": {"smbus_id": 4, "i2c_address": 0x48}
}
# ... 其他配置
```

### 2. 数据采集

#### 方式一：直接采集 (推荐)

使用 `handcap.py` 直接读取传感器数据并保存。

```bash
python handcap.py
```
数据将默认保存到 `data/handcap_{timestamp}` 目录下。

#### 方式二：ROS2 采集

如果你的环境依赖 ROS2，可以使用 `data_recorder.py`。

```bash
# 需确保 ROS2 环境已 source
python data_recorder.py
```

### 3. 数据后处理

采集完成后，可以使用 `Postprocess/` 目录下的脚本将数据转换为训练所需的格式。

*   **转换为 LeRobot 格式**:
    ```bash
    python Postprocess/_0_combine_and_transfer_data_into_lerobot.py
    ```
*   **转换为 UMI 格式**:
    ```bash
    python Postprocess/_02_combine_and_transfer_data_into_umi.py
    ```

### 4. 训练与评估

本项目采集的数据经过处理后，可直接对接主流的机器人学习训练框架。

#### 4.1 Diffusion Policy / ACT (基于 LeRobot)

本项目深度集成了 [Hugging Face LeRobot](https://github.com/huggingface/lerobot) 框架，支持 Diffusion Policy 和 ACT 算法的训练。

1.  **数据转换**:
    使用 `Postprocess/_0_combine_and_transfer_data_into_lerobot.py` 脚本将采集的原始数据转换为 LeRobot 标准数据集格式 (Hugging Face Dataset)。
    ```bash
    python Postprocess/_0_combine_and_transfer_data_into_lerobot.py \
        --data_root data/handcap_raw \
        --output_root data/lerobot_dataset \
        --time_file "20251103"
    ```
    *   `--data_root`: 原始数据存放路径。
    *   `--output_root`: 转换后的数据集输出路径。
    *   `--time_file`: 指定要处理的数据日期前缀 (如 "20251103")。

2.  **模型训练**:
    转换后的数据可直接用于 LeRobot 的训练脚本。请参考 LeRobot 官方文档进行配置和训练。

#### 4.2 Pi0 / UMI

本项目支持 [UMI (Universal Manipulation Interface)](https://umi-gripper.github.io/) 生态及 Pi0 模型的训练数据生成。

1.  **数据转换**:
    使用 `Postprocess/_02_combine_and_transfer_data_into_umi.py` 将数据转换为 UMI 兼容的 Zarr 格式 (ReplayBuffer)。
    ```bash
    python Postprocess/_02_combine_and_transfer_data_into_umi.py \
        --data_root data/handcap_raw \
        --output_root data/umi_dataset \
        --time_file "20251103"
    ```
    该脚本会生成 `combined_data.zarr` 文件，包含图像、触觉、力觉和位姿数据。

2.  **真机评估 (Pi0)**:
    在 `eval_real/` 目录下提供了针对 Pi0 模型的真机评估脚本。
    *   `eval_real_flexiv_pi0.py`: 用于在 Flexiv 机器人上评估 Pi0 策略。
    ```bash
    python eval_real/eval_real_flexiv_pi0.py \
        --policy pi0 \
        --ckpt_path /path/to/your/checkpoint \
        --robot_frequency 20
    ```
    *   `--policy`: 策略名称 (如 `pi0`)。
    *   `--ckpt_path`: 模型权重路径。
    *   `--robot_frequency`: 机器人控制频率 (Hz)。

### 5. 传感器测试与可视化

在 `test_sensor/` 目录下提供了单独测试各个传感器的脚本，用于排查硬件问题。

*   测试力传感器: `python test_sensor/read_force_v2.py`
*   测试角度传感器: `python test_sensor/read_as5600.py`
*   可视化 Vive Tracker: `python test_sensor/vive_coordinate_visualize.py`

## ⚠️ 注意事项

*   **权限**: 访问 `/dev/video*` 或 I2C 设备通常需要用户在 `video` 或 `i2c` 用户组，或者使用 `sudo` 运行。
*   **时间同步**: 对于多传感器融合，时间同步至关重要。建议配置系统时间同步服务 (如 `systemd-timesyncd`)。
*   **依赖**: 请确保安装了项目所需的 Python 依赖库 (如 `opencv-python`, `numpy`, `torch`, `smbus2` 等) 以及 ROS2 相关库 (如果使用 ROS2 采集)。

## 📅 时间校准 (Time Align)

为了保证数据的时间戳准确，建议在采集前检查系统时间设置：

```bash
timedatectl status
sudo systemctl enable --now systemd-timesyncd
sudo timedatectl set-timezone Asia/Shanghai
```