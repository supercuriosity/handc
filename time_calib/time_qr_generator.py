


import cv2
import qrcode
from PIL import Image
import numpy as np
import datetime

def main():
    """
    生成一个包含当前精确时间（ISO 8601格式）的动态二维码并显示。
    """
    print("正在启动时间二维码生成器...")
    print("窗口会显示实时时间的二维码。按 'q' 键退出。")

    while True:
        # 1. 获取当前本地时间，包含时区信息 (ISO 8601 格式)
        # 这种格式非常标准，例如: '2025-09-17T15:53:16.123456+08:00'
        now = datetime.datetime.now(datetime.timezone.utc).astimezone()
        timestamp_str = now.isoformat()

        # 2. 生成二维码
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(timestamp_str)
        qr.make(fit=True)

        # 3. 将二维码转换为OpenCV可以显示的图像格式
        qr_img_pil = qr.make_image(fill_color="black", back_color="white").convert('RGB')
        qr_img_cv = np.array(qr_img_pil)
        
        # 在图像上显示当前时间字符串，方便核对
        cv2.putText(
            qr_img_cv,
            now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], # 显示到毫秒
            (50, 450), # 位置
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, # 字体大小
            (0, 0, 255), # 颜色 (红色)
            2 # 粗细
        )

        # 4. 显示图像
        cv2.imshow('Time QR Code Generator (Source)', qr_img_cv)

        # 按 'q' 退出循环
        if cv2.waitKey(50) & 0xFF == ord('q'): # 每50毫秒刷新一次
            break

    cv2.destroyAllWindows()
    print("程序已退出。")

if __name__ == '__main__':
    main()