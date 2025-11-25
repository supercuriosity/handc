import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from PIL import Image as PILImage


class RightImagePublisher(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.publisher_ = self.create_publisher(Image, '/camera/tactile_right', 10)
        timer_period = 0.05  # 20 FPS
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.cap = cv2.VideoCapture("/dev/video2") # 0 代表第一个摄像头
        if not self.cap.isOpened():
            self.get_logger().error('Could not open video capture device at index 0')
            rclpy.shutdown()

        self.bridge = CvBridge()
        self.get_logger().info('Right Tactile Camera node has been started and is publishing.')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # 将OpenCV图像转换为ROS Image消息
            height, width = frame.shape[:2]
            crop_width = int(width * 3 / 5)
            crop_height = int(height * 3 / 5)
            start_x = (width - crop_width) // 2
            start_y = (height - crop_height) // 2
            cropped_frame = frame[start_y:start_y + crop_height,
                                  start_x:start_x + crop_width]

            # 2. Convert to PIL for easier resizing and cropping (BGR -> RGB)
            color_image_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(color_image_rgb)

            # 3. Center square crop
            w, h = pil_image.size
            size = min(w, h)
            left = (w - size) // 2
            top = (h - size) // 2
            right = left + size
            bottom = top + size
            pil_image = pil_image.crop((left, top, right, bottom))

            # 4. Resize to the final 224x224 resolution
            pil_image = pil_image.resize((224, 224), PILImage.LANCZOS)

            # 5. Convert back to NumPy array (it's now RGB)
            processed_array_rgb = np.array(pil_image)

            # 6. Convert back to BGR for ROS/cv_bridge
            final_frame = cv2.cvtColor(processed_array_rgb, cv2.COLOR_RGB2BGR)
            ros_image_msg = self.bridge.cv2_to_imgmsg(final_frame, "bgr8")
            self.publisher_.publish(ros_image_msg)
        else:
            self.get_logger().warn('Failed to capture frame.')

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    image_publisher = RightImagePublisher()
    try:
        rclpy.spin(image_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        image_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()