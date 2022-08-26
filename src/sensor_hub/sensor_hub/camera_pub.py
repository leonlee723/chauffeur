''' kitti image to RViz '''
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

DATA_PATH = '/volume/data/kitti/RawData/2011_09_26/2011_09_26_drive_0005_sync'

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('camera_publisher')
        # self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.publisher_ = self.create_publisher(Image, 'camera_topic', 10)
        timer_period = 0.1 # kitti 10frame/s
        self.bridge = CvBridge()
        self.timer = self.create_timer(timer_period, self.timer_callback)
        # self.i = 0
    
    def timer_callback(self):
        img = cv2.imread(os.path.join(DATA_PATH, 'image_02/data/%010d.png'%0))

        self.publisher_.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))
        # self.get_logger().info('Publishing: "%s"' % msg.data)
        self.get_logger().info('Publishing: image')
        # self.i += 1

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()