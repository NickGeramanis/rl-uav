#!/usr/bin/env python
import math
import random

import numpy as np
import rospy
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist, Point, Pose, PoseStamped, Quaternion
from hector_uav_msgs.srv import EnableMotors
from sensor_msgs.msg import LaserScan
from src.spaces.box import Box
from src.spaces.discrete import Discrete
from std_msgs.msg import Header
import tf
from env import Env


class NavigationEnv(Env):
    __N_ACTIONS = 3
    __FORWARD = 0
    __YAW_RIGHT = 1
    __YAW_LEFT = 2

    __WAIT_TIME = rospy.Duration.from_sec(8)
    __STEP_DURATION = rospy.Duration.from_sec(0.4)

    __COLLISION_REWARD = -200
    __FORWARD_REWARD = +5
    __YAW_REWARD = -0.5

    __FLYING_ALTITUDE = 1.5  # m
    __FORWARD_LINEAR_VELOCITY = 0.5  # m/s
    __YAW_LINEAR_VELOCITY = 0.1  # m/s
    __YAW_ANGULAR_VELOCITY = 0.5  # rad/s
    __COLISION_THRESHOLD = 0.4  # m
    __INIT_ALTITUDE = 4  # m

    __VELOCITY_STANDARD_DEVIATION = 0.01

    __MEASUREMENTS = (180, 360, 540, 720, 900)

    __N_OBSERVATIONS = len(__MEASUREMENTS)

    __SPAWN_AREA1 = (
        ((-9, -9), (-9, 9)),
        ((-9, 9), (9, 9)),
        ((9, 9), (0, 9)),
        ((0, 9), (0, 0)),
        ((0, 0), (-9, 0)),
        ((-9, 0), (-9, -9))
    )

    __SPAWN_AREA2 = (
        ((-0.2, -0.2), (-3.2, 3.1)),
        ((-0.2, -9.2), (3.1, 3.1)),
        ((-9.2, -9.2), (3.1, 12.4)),
        ((-9.2, 6.1), (12.4, 12.4)),
        ((6.1, 6.1), (12.4, 3)),
        ((6.1, 9.2), (3, 3)),
        ((9.2, 9.2), (3, -3)),
        ((9.2, 6.1), (-3, -3)),
        ((6.1, 6.1), (-3, -12.4)),
        ((6.1, -9.2), (-12.4, -12.4)),
        ((-9.2, -9.2), (-12.4, -3.2)),
        ((-9.2, -0.2), (-3.2, -3.2))
    )

    __SPAWN_AREA3 = (
        ((-4.7, 4.6), (-9.3, -9.3)),
        ((4.6, 4.6), (-9.3, -15.3)),
        ((4.6, 13.8), (-15.2, -15.2)),
        ((13.8, 13.8), (-15.2, -9.3)),
        ((13.8, 20), (-9.3, -9.3)),
        ((20, 20), (-9.3, -6.1)),
        ((20, 0), (-6.1, 13.8)),
        ((-20, 0), (-6.1, 13.8)),
        ((-20, -20), (-6.1, -9.3)),
        ((-20, -13.9), (-9.3, -9.3)),
        ((-13.9, -13.9), (-9.3, -15.2)),
        ((-13.9, -4.7), (-15.2, -15.2)),
        ((-4.7, -4.7), (-15.2, -9.3))
    )

    __SPAWN_AREAS = (__SPAWN_AREA1, __SPAWN_AREA2, __SPAWN_AREA3)

    __spawn_area = None
    __ranges = None
    __range_max = None
    __range_min = None
    __cmd_vel_pub = None
    __command_pose_pub = None
    __action_space = None
    __observation_space = None

    def __init__(self, track_id=1):
        if track_id in range(1, len(self.__SPAWN_AREAS) + 1):
            self.__spawn_area = self.__SPAWN_AREAS[track_id - 1]
        else:
            e = ValueError('Invalid track id '
                           '{} ({})'.format(track_id, type(track_id)))
            rospy.logerr(e)

        self.__cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.__command_pose_pub = rospy.Publisher(
            'command/pose', PoseStamped, queue_size=10)

        rospy.Subscriber('scan', LaserScan, self.__laser_scan_callback)

        while not self.__enable_motors():
            pass

        while self.__ranges is None:
            pass

        self.__action_space = Discrete(self.__N_ACTIONS)

        high = np.array(self.__N_OBSERVATIONS * [self.__range_max],
                        dtype=np.float32)
        low = np.array(self.__N_OBSERVATIONS * [self.__range_min],
                       dtype=np.float32)
        self.__observation_space = Box(low=low, high=high,
                                       shape=(self.__N_OBSERVATIONS,),
                                       dtype=np.float32)

    def __perform_action(self, action):
        vel_msg = Twist()

        vel_msg.linear.x = random.gauss(0, self.__VELOCITY_STANDARD_DEVIATION)
        vel_msg.angular.z = random.gauss(0, self.__VELOCITY_STANDARD_DEVIATION)

        if action == self.__FORWARD:
            vel_msg.linear.x += self.__FORWARD_LINEAR_VELOCITY
        elif action == self.__YAW_RIGHT:
            vel_msg.linear.x += self.__YAW_LINEAR_VELOCITY
            vel_msg.angular.z += self.__YAW_ANGULAR_VELOCITY
        elif action == self.__YAW_LEFT:
            vel_msg.linear.x += self.__YAW_LINEAR_VELOCITY
            vel_msg.angular.z -= self.__YAW_ANGULAR_VELOCITY

        self.__cmd_vel_pub.publish(vel_msg)

    def __fly_to(self, x, y, z, roll, pitch, yaw):
        position = Point(x, y, z)

        quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        orientation = Quaternion(
            quaternion[0], quaternion[1], quaternion[2], quaternion[3])

        pose = Pose(position, orientation)
        header = Header(frame_id='world')
        pose_stamped = PoseStamped(header, pose)

        self.__command_pose_pub.publish(pose_stamped)

    def __enable_motors(self):
        rospy.wait_for_service('enable_motors')
        try:
            enable_motors = rospy.ServiceProxy('enable_motors', EnableMotors)
            response = enable_motors(True)
            return response.success
        except rospy.ServiceException as e:
            rospy.logerr(e)

        return False

    def __laser_scan_callback(self, laser_scan):
        if self.__ranges is None:
            self.__range_max = laser_scan.range_max
            self.__range_min = laser_scan.range_min
            self.__ranges = np.empty((len(laser_scan.ranges),))

        for i in range(len(laser_scan.ranges)):
            if (laser_scan.range_min <= laser_scan.ranges[i]
                    <= laser_scan.range_max):
                self.__ranges[i] = laser_scan.ranges[i]

    def __collision_occured(self):
        if self.__ranges is None:
            return False

        return bool((self.__ranges < self.__COLISION_THRESHOLD).any())

    def __reset_world(self):
        rospy.wait_for_service('gazebo/reset_world')
        try:
            reset_env = rospy.ServiceProxy('gazebo/reset_world', Empty)
            reset_env()
        except rospy.ServiceException as e:
            rospy.logerr(e)

    def reset(self):
        self.__reset_world()
        rospy.sleep(self.__WAIT_TIME)

        area = random.choice(self.__spawn_area)
        x = random.uniform(area[0][0], area[0][1])
        y = random.uniform(area[1][0], area[1][1])
        yaw = random.uniform(-math.pi, math.pi)

        self.__fly_to(0, 0, self.__INIT_ALTITUDE, 0, 0, 0)
        rospy.sleep(self.__WAIT_TIME)
        self.__fly_to(x, y, self.__INIT_ALTITUDE, 0, 0, yaw)
        rospy.sleep(self.__WAIT_TIME)
        self.__fly_to(x, y, self.__FLYING_ALTITUDE, 0, 0, yaw)
        rospy.sleep(self.__WAIT_TIME)

        observation = list(self.__ranges)

        return observation

    def step(self, action):
        if not self.__action_space.contains(action):
            e = ValueError('Invalid action '
                           '{} ({})'.format(action, type(action)))
            rospy.logerr(e)

        self.__perform_action(action)
        rospy.sleep(self.__STEP_DURATION)

        observation = list(self.__ranges)

        done = self.__collision_occured()
        reward = 0.0

        if done:
            reward = self.__COLLISION_REWARD
        elif action == self.__FORWARD:
            reward = self.__FORWARD_REWARD
        elif action in (self.__YAW_LEFT, self.__YAW_RIGHT):
            reward = self.__YAW_REWARD

        return observation, reward, done, []

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass

    @property
    def action_space(self):
        return self.__action_space

    @property
    def observation_space(self):
        return self.__observation_space
