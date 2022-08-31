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
from collections import deque
import numpy as np

from sensor_hub.data_utils import *
from sensor_hub.publish_utils import *
from sensor_hub.kitti_util import *


DATA_PATH = '/volume/data/kitti/RawData/2011_09_26/2011_09_26_drive_0005_sync'

class Object():
    def __init__(self):
        self.locations = deque(maxlen=20)
    
    def update(self, displacement, yaw_change):
        for i in range(len(self.locations)):
            x0, y0 = self.locations[i]
            x1 = x0 * np.cos(yaw_change) + y0 * np.sin(yaw_change) - displacement
            y1 = -x0 * np.sin(yaw_change) + y0 * np.cos(yaw_change)
            self.locations[i] = np.array([x1, y1])
        self.locations.appendleft(np.array([0,0]))

    def reset(self):
        self.locations = deque(maxlen=20)

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('camera_publisher')
        # self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.cam_publisher = self.create_publisher(Image, 'camera_topic', 10)
        self.point_publisher = self.create_publisher(PointCloud2, 'point_topic', 10)
        self.ego_publisher = self.create_publisher(MarkerArray, 'ego_car_topic', 10)
        self.imu_publisher = self.create_publisher(Imu, 'imu_topic', 10)
        self.gps_publisher = self.create_publisher(NavSatFix, 'gps_topic', 10)
        self.box3d_publisher = self.create_publisher(MarkerArray,'box3d_topic', 10)
        self.loc_publisher = self.create_publisher(MarkerArray,'loc_topic', 10)
        timer_period = 0.1 # kitti 10frame/s
        self.bridge = CvBridge()
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.frame = 0
        self.df_tracking = read_tracking('/volume/data/kitti/training/label_02/0000.txt')
        self.calib =Calibration('/volume/data/kitti/RawData/2011_09_26/', from_video=True)
        
        self.ego_car_loc = Object()
        self.prev_imu_data = None 

    def compute_3d_box_cam2(self,h,w,l,x,y,z,yaw):
        R = np.array([[np.cos(yaw), 0, np.sin(yaw)],[0,1,0],[-np.sin(yaw), 0, np.cos(yaw)]])
        x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
        y_corners = [0,0,0,0,-h,-h,-h,-h]
        z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
        corners_3d_cam2 = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
        corners_3d_cam2 += np.vstack([x,y,z])
        return corners_3d_cam2

    def timer_callback(self):
        # img = cv2.imread(os.path.join(DATA_PATH, 'image_02/data/%010d.png'%self.frame))
        img = read_camera(os.path.join(DATA_PATH, 'image_02/data/%010d.png'%self.frame))
        boxes2d = np.array(self.df_tracking[self.df_tracking.frame==self.frame][['bbox_left','bbox_top','bbox_right','bbox_bottom']])
        types = np.array(self.df_tracking[self.df_tracking.frame==self.frame]['type'])
        track_ids = np.array(self.df_tracking[self.df_tracking.frame==self.frame]['track_id'])
        boxes3d = np.array(self.df_tracking[self.df_tracking.frame==self.frame][['height','width','length','pos_x','pos_y','pos_z','rot_y']])
        # print(boxes3d)
        corners_3d_velos = []
        # print("boxes lens:"+str(len(boxes3d)))
        for box3d in boxes3d:
            corners_3d_cam2 = self.compute_3d_box_cam2(*box3d)
            corners_3d_velo = self.calib.project_rect_to_velo(corners_3d_cam2.T)
            corners_3d_velos += [corners_3d_velo]
        # print("corners_3d_velos lens:"+str(len(corners_3d_velos)))
        # points = np.fromfile(os.path.join(DATA_PATH, 'velodyne_points/data/%010d.bin'%self.frame), dtype=np.float32).reshape(-1,4)
        points = read_point_cloud(os.path.join(DATA_PATH, 'velodyne_points/data/%010d.bin'%self.frame))
        imu_gps_data = read_imu(os.path.join(DATA_PATH, 'oxts/data/%010d.txt'%self.frame))

        if self.prev_imu_data is not None:
            displacement = 0.1*np.linalg.norm(imu_gps_data[['vf','vl']])
            yaw_change = float(imu_gps_data.yaw - self.prev_imu_data.yaw)    
            self.ego_car_loc.update(displacement, yaw_change)

        self.prev_imu_data = imu_gps_data
        
        # self.cam_publisher.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))
        publish_camera(self.cam_publisher, self.bridge, img, boxes2d, types)
        # self.point_publisher.publish(self.point_cloud(points[:,:3], 'map'))
        publish_point_cloud(self.point_publisher, points)
        publish_ego_car(self.ego_publisher)
        publish_imu(self.imu_publisher,imu_gps_data)
        publish_gps(self.gps_publisher,imu_gps_data)
        publish_3dbox(self.box3d_publisher, corners_3d_velos,types,track_ids)
        publish_loc(self.loc_publisher, self.ego_car_loc.locations)
        # self.get_logger().info('Publishing: "%s"' % msg.data)
        self.get_logger().info('Publishing: image')
        # self.get_logger().info(os.path.join(get_package_share_directory('sensor_hub'), 'meshes'))
        self.frame += 1
        #一共154张照片，0-153，如果小于154，取余数与frame相同，如果frame大于154，则从0开始
        #self.frame %= 154  
        if self.frame == 154:
            self.frame = 0
            self.ego_car_loc.reset()


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()