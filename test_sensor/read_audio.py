
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

# --- 配置 ---
# 根据 `python3 -m sounddevice` 的最新输出结果更新此值
DEVICE_INDEX = 2       

# 音频参数
SAMPLE_RATE = 44100      # 采样率 (Hz)
CHANNELS = 1             # 通道数 (推荐单声道以避免兼容性问题)
DURATION = 5             # 录制时长 (秒)
OUTPUT_FILENAME = "output_from_index.wav"

# 检查设备有效性
try:
    device_info = sd.query_devices(DEVICE_INDEX, 'input')
    print(f"✅ 成功找到设备索引 {DEVICE_INDEX}: {device_info['name']}")
    
    # 尝试使用设备推荐的采样率，如果失败则使用默认值
    if device_info['default_samplerate'] > 0:
        SAMPLE_RATE = int(device_info['default_samplerate'])
        print(f"   使用设备推荐采样率: {SAMPLE_RATE} Hz")
    else:
        print(f"   设备未提供推荐采样率，使用默认值: {SAMPLE_RATE} Hz")
        
except (ValueError, IndexError) as e:
    print(f"❌ 错误: 设备索引 {DEVICE_INDEX} 无效或不是一个输入设备。")
    print("   请再次运行 'python3 -m sounddevice' 检查正确的索引。")
    exit()


print(f"\n准备录音... 将从设备索引 {DEVICE_INDEX} 录制 {DURATION} 秒。")

# 使用 sd.rec() 直接录制
# dtype='int16' 对应 16-bit PCM 格式
myrecording = sd.rec(int(DURATION * SAMPLE_RATE), 
                     samplerate=SAMPLE_RATE, 
                     channels=CHANNELS, 
                     device=DEVICE_INDEX,
                     dtype='int16')

# 等待录音完成
sd.wait()


# print(myrecording[10000:15000])
# exit()
print("录音完成！")

# myrecording 现在是一个 NumPy 数组，包含了原始的音频数据。
# 为了验证，我们将其保存为 .wav 文件。
print(f"💾 正在保存为 '{OUTPUT_FILENAME}'...")
wav.write(OUTPUT_FILENAME, SAMPLE_RATE, myrecording)
print("保存成功！")


# === 持续录制 直至暂停 ===
# import numpy as np
# import sounddevice as sd
# import scipy.io.wavfile as wav
# import queue

# # --- 配置 ---
# # 根据 `python3 -m sounddevice` 的最新输出结果更新此值
# DEVICE_INDEX = 1        

# # 音频参数
# SAMPLE_RATE = 44100      # 采样率 (Hz)
# CHANNELS = 1             # 通道数 (推荐单声道以避免兼容性问题)
# BLOCKSIZE = 1024         # 每次从流中读取的帧数
# DTYPE = 'int16'          # 数据类型
# OUTPUT_FILENAME = "continuous_recording.wav"

# # 检查设备有效性
# try:
#     device_info = sd.query_devices(DEVICE_INDEX, 'input')
#     print(f"✅ 成功找到设备索引 {DEVICE_INDEX}: {device_info['name']}")
    
#     # 尝试使用设备推荐的采样率，如果失败则使用默认值
#     if device_info['default_samplerate'] > 0:
#         SAMPLE_RATE = int(device_info['default_samplerate'])
#         print(f"   使用设备推荐采样率: {SAMPLE_RATE} Hz")
#     else:
#         print(f"   设备未提供推荐采样率，使用默认值: {SAMPLE_RATE} Hz")
        
# except (ValueError, IndexError) as e:
#     print(f"❌ 错误: 设备索引 {DEVICE_INDEX} 无效或不是一个输入设备。")
#     print("   请再次运行 'python3 -m sounddevice' 检查正确的索引。")
#     exit()

# # 创建一个队列来安全地在线程间传递音频数据
# q = queue.Queue()

# def audio_callback(indata, frames, time, status):
#     """这是一个回调函数，音频设备每准备好一块数据就会调用它。"""
#     if status:
#         print(status)
#     q.put(indata.copy())

# # --- 主程序 ---
# try:
#     # 使用with语句确保流在结束时能够被正确关闭
#     with sd.InputStream(samplerate=SAMPLE_RATE, 
#                          device=DEVICE_INDEX, 
#                          channels=CHANNELS, 
#                          dtype=DTYPE,
#                          callback=audio_callback):
        
#         print("\n🔴 录音已开始... 按下 Ctrl+C 停止录音。")
        
#         # 主循环：从队列中获取数据并存入列表
#         recording_data = []
#         while True:
#             recording_data.append(q.get())

# except KeyboardInterrupt:
#     print("\n⏹️ 录音已停止。")
#     # 当用户按下 Ctrl+C 时，循环会在这里中断

# except Exception as e:
#     print(f"发生错误: {e}")

# # --- 保存文件 ---
# if not recording_data:
#     print("没有录制到任何音频数据。")
# else:
#     print(f"💾 正在处理并保存为 '{OUTPUT_FILENAME}'...")
    
#     # 将列表中的所有Numpy数组块合并成一个大的数组
#     final_recording = np.concatenate(recording_data, axis=0)
    
#     # 写入 .wav 文件
#     wav.write(OUTPUT_FILENAME, SAMPLE_RATE, final_recording)
    
#     print("保存成功！")