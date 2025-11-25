import cv2
import time
from pyzbar import pyzbar

frame = cv2.imread("/root/workspace/handcap/Postprocess/image.png", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

if frame is None:
    raise FileNotFoundError("无法读取指定图像，请检查路径或文件权限。")
decoded_objects = pyzbar.decode(thresh)
# detector = cv2.QRCodeDetector()
# data, pts, _ = detector.detectAndDecode(frame)
# print(data, pts)
print(decoded_objects)