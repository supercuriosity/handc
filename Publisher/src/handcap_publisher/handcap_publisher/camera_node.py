import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        timer_period = 0.05  # 20 FPS
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.cap = cv2.VideoCapture("/dev/video4") # 0 代表第一个摄像头
        if not self.cap.isOpened():
            self.get_logger().error('Could not open video capture device at index 0')
            rclpy.shutdown()

        self.bridge = CvBridge()
        self.get_logger().info('Camera node has been started and is publishing.')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # 将OpenCV图像转换为ROS Image消息
            ros_image_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.publisher_.publish(ros_image_msg)
        else:
            self.get_logger().warn('Failed to capture frame.')

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher()
    try:
        rclpy.spin(image_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        image_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()