from cgi import test
from math import dist
import os
from distutils.archive_util import make_archive
from re import T
# from std_msgs.msg import Header
# from cv_bridge import CvBridge
import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs
from sensor_msgs.msg import Imu, NavSatFix
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
import rclpy as rclpy
from rclpy.duration import Duration
from transforms3d._gohlketransforms import quaternion_from_euler
from ament_index_python.packages import get_package_share_directory
import cv2

FRAME_ID = 'map'
DETECTION_COLOR_DICT = {'Car':(255,255,0), 'Pedestrian':(0,226,255), 'Cyclist':(141,40,255)}

LINES  = [[0, 1], [1, 2], [2, 3], [3, 0]]  # Lower plane parallel to Z=0 plane
LINES += [[4, 5], [5, 6], [6, 7], [7, 4]]  # Upper plane parallel to Z=0 plane
LINES += [[0, 4], [1, 5], [2, 6], [3, 7]]  # Connections between upper and lower planes
LINES += [[4, 1], [5, 0]] # front face X

def publish_camera(cam_pub, bridge, image, boxes, types):
    for typ, box in zip(types,boxes):
        top_left = int(box[0]), int(box[1])
        bottom_right = int(box[2]), int(box[3])
        cv2.rectangle(image,top_left,bottom_right,DETECTION_COLOR_DICT[typ],2)
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
    marker_array = MarkerArray()

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

    marker_array.markers.append(marker)

    mesh_marker = Marker()
    mesh_marker.header.frame_id = FRAME_ID
    
    mesh_marker.id = -1
    mesh_marker.lifetime = Duration().to_msg()
    mesh_marker.type = Marker.MESH_RESOURCE
    # mesh_marker.mesh_resource = "package://sensor_hub/meshes/4096-MicroWheelsG.dae"
    mesh_marker.mesh_resource = "file://"+os.path.join(get_package_share_directory('sensor_hub'), 'meshes','4096-MicroWheelsG.dae')

    mesh_marker.pose.position.x = -2.0
    mesh_marker.pose.position.y = 0.0
    mesh_marker.pose.position.z = -1.7

    q = quaternion_from_euler(-np.pi/2, np.pi, -np.pi/2) # roll, pitch and yaw
    mesh_marker.pose.orientation.x = q[0]
    mesh_marker.pose.orientation.y = q[1]
    mesh_marker.pose.orientation.z = q[2]
    mesh_marker.pose.orientation.w = q[3]

    mesh_marker.color.r = 1.0
    mesh_marker.color.g = 1.0
    mesh_marker.color.b = 1.0
    mesh_marker.color.a = 1.0

    mesh_marker.scale.x = 30.0
    mesh_marker.scale.y = 30.0
    mesh_marker.scale.z = 30.0
    marker_array.markers.append(mesh_marker)

    ego_car_pub.publish(marker_array)

def publish_imu(imu_pub, imu_data):
    imu = Imu()
    imu.header.frame_id = FRAME_ID
    
    q = quaternion_from_euler(float(imu_data.roll), float(imu_data.pitch), float(imu_data.yaw)) # roll, pitch and yaw
    imu.orientation.x = q[0]
    imu.orientation.y = q[1]
    imu.orientation.z = q[2]
    imu.orientation.w = q[3]

    imu.linear_acceleration.x = float(imu_data.af)
    imu.linear_acceleration.y = float(imu_data.al)
    imu.linear_acceleration.z = float(imu_data.au)

    imu.angular_velocity.x = float(imu_data.wf)
    imu.angular_velocity.y = float(imu_data.wl)
    imu.angular_velocity.z = float(imu_data.wu)

    imu_pub.publish(imu)

def publish_gps(gps_pub, gps_data):
    gps = NavSatFix()
    gps.header.frame_id = FRAME_ID

    gps.latitude = float(gps_data.lat)
    gps.longitude = float(gps_data.lon)
    gps.altitude = float(gps_data.alt)

    gps_pub.publish(gps)

def publish_3dbox(box3d_pub, corners_3d_velos, types, track_ids):
    marker_array = MarkerArray()
    for i, corners_3d_velo in enumerate(corners_3d_velos):
        marker = Marker()
        marker.header.frame_id = FRAME_ID
        
        marker.id = i
        marker.action = Marker.ADD
        marker.lifetime = Duration(seconds=0.1).to_msg()
        marker.type = Marker.LINE_LIST

        b ,g ,r = DETECTION_COLOR_DICT[types[i]]
        marker.color.r = r/255.0
        marker.color.g = g/255.0
        marker.color.b = b/255.0

        marker.color.a = 1.0
        marker.scale.x = 0.1

        marker_points = []
        for l in LINES:
            p1 = corners_3d_velo[l[0]]
            point1 = Point()
            point1.x = p1[0]
            point1.y = p1[1]
            point1.z = p1[2]
            marker.points.append(point1)
            p2 = corners_3d_velo[l[1]]
            point2 = Point()
            point2.x = p2[0]
            point2.y = p2[1]
            point2.z = p2[2]
            marker.points.append(point2)

        marker_array.markers.append(marker)

        text_marker = Marker()
        text_marker.header.frame_id = FRAME_ID
        
        text_marker.id = i + 1000
        text_marker.action = Marker.ADD
        text_marker.lifetime = Duration(seconds=0.1).to_msg()
        text_marker.type = Marker.TEXT_VIEW_FACING

        p4 = np.mean(corners_3d_velo, axis=0)

        text_marker.pose.position.x = p4[0]
        text_marker.pose.position.y = p4[1]
        text_marker.pose.position.z = p4[2] + 1

        text_marker.text = str(track_ids[i])

        text_marker.scale.x = 1.0
        text_marker.scale.y = 1.0
        text_marker.scale.z = 1.0

        b ,g ,r = DETECTION_COLOR_DICT[types[i]]
        text_marker.color.r = r/255.0
        text_marker.color.g = g/255.0
        text_marker.color.b = b/255.0
        text_marker.color.a = 1.0        

        marker_array.markers.append(text_marker)
    
    box3d_pub.publish(marker_array)

def publish_loc(loc_pub, tracker, centers):
    marker_array = MarkerArray()

    for track_id in centers:

        marker = Marker()
        marker.header.frame_id = FRAME_ID
        
        marker.action = Marker.ADD
        marker.lifetime = Duration(seconds=0.1).to_msg()
        marker.type = Marker.LINE_STRIP
        marker.id = int(track_id)

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.scale.x = 0.2
        marker.points = []
        for p in tracker[track_id].locations:
            # print(p)
            point = Point()
            point.x = float(p[0])
            point.y = float(p[1])
            point.z = float(0.0)
            marker.points.append(point)

        marker_array.markers.append(marker)
    loc_pub.publish(marker_array)

def publish_dist(dist_pub, minPQDs):
    marker_array = MarkerArray()

    for i, (minP, minQ, minD) in enumerate(minPQDs):
        marker = Marker()
        marker.header.frame_id = FRAME_ID

        marker.action = Marker.ADD
        marker.lifetime = Duration(seconds=0.1).to_msg()
        marker.type = Marker.LINE_STRIP
        marker.id = i

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.5
        marker.scale.x = 0.1

        marker.points = []
        point = Point()
        point.x = float(minP[0])
        point.y = float(minP[1])
        point.z = float(0.0)
        marker.points.append(point)
        point2 = Point()
        point2.x = float(minQ[0])
        point2.y = float(minQ[1])
        point2.z = float(0.0)
        marker.points.append(point2)

        marker_array.markers.append(marker)

        text_marker = Marker()
        text_marker.header.frame_id = FRAME_ID
        
        text_marker.id = i + 1000
        text_marker.action = Marker.ADD
        text_marker.lifetime = Duration(seconds=0.1).to_msg()
        text_marker.type = Marker.TEXT_VIEW_FACING

        p = (minP + minQ) / 2.0
        text_marker.pose.position.x = p[0]
        text_marker.pose.position.y = p[1]
        text_marker.pose.position.z = 0.0

        text_marker.text = '%.2f'%minD

        text_marker.scale.x = 1.0
        text_marker.scale.y = 1.0
        text_marker.scale.z = 1.0

        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 0.8
        marker_array.markers.append(text_marker)

    dist_pub.publish(marker_array)