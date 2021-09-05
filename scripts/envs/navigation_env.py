#!/usr/bin/env python
import math
import random

import numpy as np
import rospy
import std_srvs
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
from hector_uav_msgs.msg import *
from hector_uav_msgs.srv import *
from sensor_msgs.msg import *
from spaces.box import Box
from spaces.discrete import Discrete
from std_msgs.msg import *
from std_srvs.srv import *
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from env import Env


class NavigationEnv(Env):
    N_ACTIONS = 3
    FORWARD = 0
    YAW_RIGHT = 1
    YAW_LEFT = 2

    STEP_DURATION = rospy.Duration.from_sec(0.4)

    COLLISION_REWARD = -200
    FORWARD_REWARD = +5
    YAW_REWARD = -0.5
    STEP_REWARD = 0

    FLYING_ALTITUDE = 1.5  # m
    FORWARD_LINEAR_VELOCITY = 0.5  # m/s
    YAW_LINEAR_VELOCITY = 0.1  # m/s
    YAW_ANGULAR_VELOCITY = 0.5  # rad/s
    WALL_DISTANCE_THRESHOLD = 0.4  # m
    INIT_ALTITUDE = 4  # m

    MAX_ACTIONS = 500

    VELOCITY_STANDARD_DEVIATION = 0.01  # m/s

    MEASUREMENTS = (180, 360, 540, 720, 900)

    N_OBSERVATIONS = len(MEASUREMENTS)

    # track1
    SPAWNABLE_AREA = (
        ((-9, -9), (-9, 9)),
        ((-9, 9), (9, 9)),
        ((9, 9), (0, 9)),
        ((0, 9), (0, 0)),
        ((0, 0), (-9, 0)),
        ((-9, 0), (-9, -9))
    )
    '''
    # track2
    SPAWNABLE_AREA = (
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

    # track3
    SPAWNABLE_AREA = (
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
    '''

    def __init__(self):
        self.ranges = None
        self.range_max = None
        self.range_min = None
        self.total_actions = 0

        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.command_pose_pub = rospy.Publisher(
            'command/pose', PoseStamped, queue_size=10)

        rospy.Subscriber('scan', LaserScan, self.laser_scan_callback)

        while not self.enable_motors():
            pass

        while self.ranges is None:
            pass

        self.action_space = Discrete(self.N_ACTIONS)

        high = np.array(
            self.N_OBSERVATIONS * [self.range_max], dtype=np.float32)
        low = np.array(
            self.N_OBSERVATIONS * [self.range_min], dtype=np.float32)
        self.observation_space = Box(
            low=low, high=high, shape=(self.N_OBSERVATIONS,), dtype=np.float32)

    def perform_action(self, action):
        vel_msg = Twist()

        linear_velocity_noise = random.gauss(
            0, self.VELOCITY_STANDARD_DEVIATION)
        angular_velocity_noise = random.gauss(
            0, self.VELOCITY_STANDARD_DEVIATION)
        if action == self.FORWARD:
            vel_msg.linear.x = (self.FORWARD_LINEAR_VELOCITY
                                + linear_velocity_noise)
            vel_msg.angular.z = angular_velocity_noise
        elif action == self.YAW_LEFT:
            vel_msg.linear.x = self.YAW_LINEAR_VELOCITY + linear_velocity_noise
            vel_msg.angular.z = (self.YAW_ANGULAR_VELOCITY
                                 + angular_velocity_noise)
        else:
            vel_msg.linear.x = self.YAW_LINEAR_VELOCITY + linear_velocity_noise
            vel_msg.angular.z = (-self.YAW_ANGULAR_VELOCITY
                                 + angular_velocity_noise)

        self.cmd_vel_pub.publish(vel_msg)

    def fly_to(self, x, y, z, roll, pitch, yaw):
        position = Point(x, y, z)

        quaternion = quaternion_from_euler(roll, pitch, yaw)
        orientation = Quaternion(
            quaternion[0], quaternion[1], quaternion[2], quaternion[3])

        pose = Pose(position, orientation)
        header = Header(frame_id='world')
        pose_stamped = PoseStamped(header, pose)
        self.command_pose_pub.publish(pose_stamped)

    def enable_motors(self):
        rospy.wait_for_service('enable_motors')
        try:
            enable_motors = rospy.ServiceProxy('enable_motors', EnableMotors)
            response = enable_motors(True)
            return response.success
        except rospy.ServiceException as e:
            rospy.logerr(e)

        return False

    def laser_scan_callback(self, laser_scan):
        if self.ranges is None:
            self.range_max = laser_scan.range_max
            self.range_min = laser_scan.range_min
            self.ranges = np.empty((len(laser_scan.ranges),))

        for range_i in range(len(laser_scan.ranges)):
            if (laser_scan.range_min <= laser_scan.ranges[range_i] <=
                    laser_scan.range_max):
                self.ranges[range_i] = laser_scan.ranges[range_i]

    def collision_occured(self):
        if self.ranges is None:
            return False

        return np.any(self.ranges < self.WALL_DISTANCE_THRESHOLD)

    def reset_world(self):
        rospy.wait_for_service('gazebo/reset_world')
        try:
            reset_env = rospy.ServiceProxy(
                'gazebo/reset_world', std_srvs.srv.Empty)
            reset_env()
        except rospy.ServiceException as e:
            rospy.logerr(e)

    def reset(self):
        self.reset_world()
        rospy.sleep(0.5)

        area = random.choice(self.SPAWNABLE_AREA)
        x = random.uniform(area[0][0], area[0][1])
        y = random.uniform(area[1][0], area[1][1])
        yaw = random.uniform(-math.pi, math.pi)

        self.fly_to(0, 0, self.INIT_ALTITUDE, 0, 0, 0)
        rospy.sleep(8)
        self.fly_to(x, y, self.INIT_ALTITUDE, 0, 0, yaw)
        rospy.sleep(8)
        self.fly_to(x, y, self.FLYING_ALTITUDE, 0, 0, yaw)
        rospy.sleep(8)

        self.time = rospy.get_rostime()

        self.total_actions = 0
        observation = [self.ranges[range_i] for range_i in self.MEASUREMENTS]
        return observation

    def step(self, action):
        if not self.action_space.contains(action):
            rospy.logerr('Invalid action')

        action_selection_duration = rospy.get_rostime() - self.time
        if action_selection_duration.to_sec() >= self.STEP_DURATION.to_sec():
            rospy.logwarn('Large action selection time duration')

        self.perform_action(action)
        rospy.sleep(self.STEP_DURATION)
        self.time = rospy.get_rostime()

        observation = [self.ranges[range_i] for range_i in self.MEASUREMENTS]

        self.total_actions += 1

        collision_occured = self.collision_occured()

        done = collision_occured or self.total_actions == self.MAX_ACTIONS

        if collision_occured:
            reward = self.COLLISION_REWARD
        elif action == self.FORWARD:
            reward = self.FORWARD_REWARD
        else:
            reward = self.YAW_REWARD

        return observation, reward, done, []

    def render(self):
        pass
