''' kitti image to RViz '''
from unicodedata import name
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
from cv_bridge import CvBridge 
import cv2
import os
import numpy as np


DATA_PATH = '/volume/data/kitti/RawData/2011_09_26/2011_09_26_drive_0005_sync'

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('camera_publisher')
        # self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.cam_publisher = self.create_publisher(Image, 'camera_topic', 10)
        self.point_publisher = self.create_publisher(PointCloud2, 'point_topic', 10)
        timer_period = 0.1 # kitti 10frame/s
        self.bridge = CvBridge()
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.frame = 0
    
    def timer_callback(self):
        img = cv2.imread(os.path.join(DATA_PATH, 'image_02/data/%010d.png'%self.frame))
        points = np.fromfile(os.path.join(DATA_PATH, 'velodyne_points/data/%010d.bin'%self.frame), dtype=np.float32).reshape(-1,4)
        self.cam_publisher.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))
        self.point_publisher.publish(self.point_cloud(points[:,:3], 'map'))
        # self.get_logger().info('Publishing: "%s"' % msg.data)
        self.get_logger().info('Publishing: image')
        self.frame += 1
        #一共154张照片，0-153，如果小于154，取余数与frame相同，如果frame大于154，则从0开始
        self.frame %= 154  

    # reference to https://github.com/SebastianGrans/ROS2-Point-Cloud-Demo/blob/master/pcd_demo/pcd_publisher/pcd_publisher_node.py
    def point_cloud(self,points, parent_frame):
        ros_dtype = sensor_msgs.PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize

        data = points.astype(dtype).tobytes()
        fields = [sensor_msgs.PointField(
            name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate('xyz')]
        header = std_msgs.Header(frame_id=parent_frame)

        return sensor_msgs.PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 3),  # Every point consists of three float32s.
            row_step=(itemsize * 3 * points.shape[0]),
            data=data
        )

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()