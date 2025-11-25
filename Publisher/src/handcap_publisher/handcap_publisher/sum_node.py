import rclpy

from .angle_node import AnglePublisher
from .camera_node import ImagePublisher
from .left_force_node import LeftForcePublisher
from .right_force_node import RightForcePublisher

from .left_tactile_node import LeftImagePublisher
from .right_tactile_node import RightImagePublisher


def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher()
    left_tactile_publisher = LeftImagePublisher()
    right_tactile_publisher = RightImagePublisher()
    left_force_publisher = LeftForcePublisher()
    right_force_publisher = RightForcePublisher()
    angle_publisher = AnglePublisher()
    
    try:
        rclpy.spin(image_publisher)
        rclpy.spin(left_tactile_publisher)
        rclpy.spin(right_tactile_publisher)
        rclpy.spin(left_force_publisher)
        rclpy.spin(right_force_publisher)
        rclpy.spin(angle_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        image_publisher.destroy_node()
        left_tactile_publisher.destroy_node()
        right_tactile_publisher.destroy_node()
        left_force_publisher.destroy_node()
        right_force_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()