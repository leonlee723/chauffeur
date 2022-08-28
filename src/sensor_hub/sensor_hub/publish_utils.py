from distutils.archive_util import make_archive
# from std_msgs.msg import Header
# from cv_bridge import CvBridge
import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np
import rclpy as rclpy
from rclpy.duration import Duration

FRAME_ID = 'map'

def publish_camera(cam_pub, bridge, image):
    cam_pub.publish(bridge.cv2_to_imgmsg(image, "bgr8"))

def publish_point_cloud(pcl_pub, points):
    pcl_pub.publish(point_cloud(points=points[:,:3], parent_frame=FRAME_ID))

# reference to https://github.com/SebastianGrans/ROS2-Point-Cloud-Demo/blob/master/pcd_demo/pcd_publisher/pcd_publisher_node.py
def point_cloud(points, parent_frame):
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

def publish_ego_car(ego_car_pub):
    """
    Publish left and right 45 degree FOV lines and ego car model mesh
    """
    marker = Marker()
    marker.header.frame_id = FRAME_ID
    # marker.header.stamp
    
    marker.id = 0
    marker.action = Marker.ADD
    marker.lifetime = Duration().to_msg()
    marker.type = Marker.LINE_STRIP

    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.scale.x = 0.2

    marker.points = []
    point1 = Point()
    point1.x = 10.0
    point1.y = -10.0
    point1.z = 0.0
    marker.points.append(point1)
    point2 = Point()
    point2.x = 0.0
    point2.y = 0.0
    point2.z = 0.0
    marker.points.append(point2)
    point3 = Point()
    point3.x = 10.0
    point3.y = 10.0
    point3.z = 0.0
    marker.points.append(point3)

    ego_car_pub.publish(marker)


