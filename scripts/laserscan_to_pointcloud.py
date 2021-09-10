#!/usr/bin/env python
import laser_geometry.laser_geometry as lg
import rospy
from sensor_msgs.msg import LaserScan, PointCloud2


def scan_cb(msg):
    list_ranges = list(msg.ranges)
    for i in range(0, len(list_ranges)):
        if i not in MEASUREMENTS:
            list_ranges[i] = 0
        else:
            print(list_ranges[i])

    msg.ranges = tuple(list_ranges)
    pc2_msg = lp.projectLaser(msg)
    pc_pub.publish(pc2_msg)
    print('------------------------------------')


MEASUREMENTS = (180, 360, 540, 720, 900)

rospy.init_node("laserscan_to_pointcloud")
lp = lg.LaserProjection()
pc_pub = rospy.Publisher("converted_pc", PointCloud2, queue_size=1)
rospy.Subscriber("/scan", LaserScan, scan_cb, queue_size=1)
rospy.spin()
