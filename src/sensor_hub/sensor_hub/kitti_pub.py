''' kitti image to RViz '''
from unicodedata import name
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2

# import sensor_msgs.msg as sensor_msgs
# import std_msgs.msg as std_msgs
from cv_bridge import CvBridge 
import cv2
import os
import numpy as np

from sensor_hub.data_utils import *
from sensor_hub.publish_utils import *


DATA_PATH = '/volume/data/kitti/RawData/2011_09_26/2011_09_26_drive_0005_sync'

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('camera_publisher')
        # self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.cam_publisher = self.create_publisher(Image, 'camera_topic', 10)
        self.point_publisher = self.create_publisher(PointCloud2, 'point_topic', 10)
        self.ego_publisher = self.create_publisher(Marker, 'ego_car_topic', 10)
        timer_period = 0.1 # kitti 10frame/s
        self.bridge = CvBridge()
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.frame = 0
    
    def timer_callback(self):
        # img = cv2.imread(os.path.join(DATA_PATH, 'image_02/data/%010d.png'%self.frame))
        img = read_camera(os.path.join(DATA_PATH, 'image_02/data/%010d.png'%self.frame))
        # points = np.fromfile(os.path.join(DATA_PATH, 'velodyne_points/data/%010d.bin'%self.frame), dtype=np.float32).reshape(-1,4)
        points = read_point_cloud(os.path.join(DATA_PATH, 'velodyne_points/data/%010d.bin'%self.frame))
        # self.cam_publisher.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))
        publish_camera(self.cam_publisher, self.bridge, img)
        # self.point_publisher.publish(self.point_cloud(points[:,:3], 'map'))
        publish_point_cloud(self.point_publisher, points)
        publish_ego_car(self.ego_publisher)
        # self.get_logger().info('Publishing: "%s"' % msg.data)
        self.get_logger().info('Publishing: image')
        self.frame += 1
        #一共154张照片，0-153，如果小于154，取余数与frame相同，如果frame大于154，则从0开始
        self.frame %= 154  

    
def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()