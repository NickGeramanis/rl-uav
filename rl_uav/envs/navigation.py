#!/usr/bin/env python
"""This module contains the basic Navigation environment class."""
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import rospy
import tf
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Twist
from gym import Env, spaces
from hector_uav_msgs.srv import EnableMotors
from rospy.topics import Publisher
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from std_srvs.srv import Empty


@dataclass
class Euler:
    """Orientation represented with euler angles."""
    roll: float
    pitch: float
    yaw: float


class Navigation(Env):
    """The basic Navigation environment."""

    _N_ACTIONS = 3
    _FORWARD = 0
    _YAW_RIGHT = 1
    _YAW_LEFT = 2

    _WAIT_TIME = rospy.Duration.from_sec(8)
    _STEP_DURATION = rospy.Duration.from_sec(0.4)

    _COLLISION_REWARD = -200.0
    _FORWARD_REWARD = +5.0
    _YAW_REWARD = -0.5

    _FLYING_ALTITUDE = 1.5  # m
    _FORWARD_LINEAR_VELOCITY = 0.5  # m/s
    _YAW_LINEAR_VELOCITY = 0.1  # m/s
    _YAW_ANGULAR_VELOCITY = 0.5  # rad/s
    _COLLISION_THRESHOLD = 0.4  # m
    _INIT_ALTITUDE = 4  # m

    _VELOCITY_STANDARD_DEVIATION = 0.01

    _MEASUREMENTS = (180, 360, 540, 720, 900)

    _N_OBSERVATIONS = len(_MEASUREMENTS)

    _SPAWN_AREA1 = (
        ((-9, -9), (-9, 9)),
        ((-9, 9), (9, 9)),
        ((9, 9), (0, 9)),
        ((0, 9), (0, 0)),
        ((0, 0), (-9, 0)),
        ((-9, 0), (-9, -9))
    )

    _SPAWN_AREA2 = (
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

    _SPAWN_AREA3 = (
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

    _SPAWN_AREAS = (_SPAWN_AREA1, _SPAWN_AREA2, _SPAWN_AREA3)

    _spawn_area: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...]
    _ranges: Optional[np.ndarray]
    _ranges_range: Tuple[float, float]
    _cmd_vel_pub: Publisher
    _command_pose_pub: Publisher
    action_space: spaces.Discrete
    observation_space: spaces.Box

    def __init__(self, track_id: int = 1) -> None:
        if track_id in range(1, len(self._SPAWN_AREAS) + 1):
            self._spawn_area = self._SPAWN_AREAS[track_id - 1]
        else:
            raise ValueError(f'Invalid track id {track_id} ({type(track_id)})')

        self._ranges = None
        self._ranges_range = (-math.inf, math.inf)

        self._cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self._command_pose_pub = rospy.Publisher('command/pose',
                                                 PoseStamped,
                                                 queue_size=10)

        rospy.Subscriber('scan', LaserScan, self._laser_scan_callback)

        while not Navigation._enable_motors():
            pass

        while self._ranges is None:
            pass

        self.action_space = spaces.Discrete(self._N_ACTIONS)

        high = np.array(self._N_OBSERVATIONS * [self._ranges_range[0]],
                        dtype=np.float32)
        low = np.array(self._N_OBSERVATIONS * [self._ranges_range[1]],
                       dtype=np.float32)
        self.observation_space = spaces.Box(low=low,
                                            high=high,
                                            shape=(self._N_OBSERVATIONS,),
                                            dtype=np.float32)

    def _perform_action(self, action: int) -> None:
        vel_msg = Twist()

        vel_msg.linear.x = random.gauss(0, self._VELOCITY_STANDARD_DEVIATION)
        vel_msg.angular.z = random.gauss(0, self._VELOCITY_STANDARD_DEVIATION)

        if action == self._FORWARD:
            vel_msg.linear.x += self._FORWARD_LINEAR_VELOCITY
        elif action == self._YAW_RIGHT:
            vel_msg.linear.x += self._YAW_LINEAR_VELOCITY
            vel_msg.angular.z += self._YAW_ANGULAR_VELOCITY
        else:
            vel_msg.linear.x += self._YAW_LINEAR_VELOCITY
            vel_msg.angular.z -= self._YAW_ANGULAR_VELOCITY

        self._cmd_vel_pub.publish(vel_msg)

    def _fly_to(self, position: Point, euler: Euler) -> None:
        quaternion = tf.transformations.quaternion_from_euler(euler.roll,
                                                              euler.pitch,
                                                              euler.yaw)
        quaternion = Quaternion(quaternion[0],
                                quaternion[1],
                                quaternion[2],
                                quaternion[3])

        pose = Pose(position, quaternion)
        header = Header(frame_id='world')
        pose_stamped = PoseStamped(header, pose)

        self._command_pose_pub.publish(pose_stamped)

    @staticmethod
    def _enable_motors() -> bool:
        rospy.wait_for_service('enable_motors')
        try:
            enable_motors = rospy.ServiceProxy('enable_motors', EnableMotors)
            response = enable_motors(True)
            return response.success
        except rospy.ServiceException as exception:
            rospy.logerr(exception)

        return False

    def _laser_scan_callback(self, laser_scan: LaserScan) -> None:
        if self._ranges is None:
            self._ranges = np.empty(len(laser_scan.ranges))
            self._ranges_range = (laser_scan.range_max, laser_scan.range_min)

        for i, range_ in enumerate(laser_scan.ranges):
            if laser_scan.range_min <= range_ <= laser_scan.range_max:
                self._ranges[i] = range_

    def _collision_occurred(self) -> bool:
        if self._ranges is None:
            raise ValueError

        return bool((self._ranges < self._COLLISION_THRESHOLD).any())

    @staticmethod
    def _reset_world() -> None:
        rospy.wait_for_service('gazebo/reset_world')
        try:
            reset_env = rospy.ServiceProxy('gazebo/reset_world', Empty)
            reset_env()
        except rospy.ServiceException as exception:
            rospy.logerr(exception)

    def reset(self) -> List[float]:
        Navigation._reset_world()
        rospy.sleep(self._WAIT_TIME)

        area = random.choice(self._spawn_area)
        x_coordinate = random.uniform(area[0][0], area[0][1])
        y_coordinate = random.uniform(area[1][0], area[1][1])
        yaw = random.uniform(-math.pi, math.pi)

        self._fly_to(Point(0, 0, self._INIT_ALTITUDE),
                     Euler(0, 0, 0))
        rospy.sleep(self._WAIT_TIME)
        self._fly_to(Point(x_coordinate, y_coordinate, self._INIT_ALTITUDE),
                     Euler(0, 0, yaw))
        rospy.sleep(self._WAIT_TIME)
        self._fly_to(Point(x_coordinate, y_coordinate, self._FLYING_ALTITUDE),
                     Euler(0, 0, yaw))
        rospy.sleep(self._WAIT_TIME)

        if self._ranges is None:
            raise ValueError

        observation = list(self._ranges)

        return observation

    def step(self, action: int) -> Tuple[List[float], float, bool, List[str]]:
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action {action} ({type(action)})')

        self._perform_action(action)
        rospy.sleep(self._STEP_DURATION)

        if self._ranges is None:
            raise ValueError

        observation = list(self._ranges)

        done = self._collision_occurred()

        if done:
            reward = self._COLLISION_REWARD
        elif action == self._FORWARD:
            reward = self._FORWARD_REWARD
        else:
            reward = self._YAW_REWARD

        return observation, reward, done, []

    def render(self, mode="human"):
        """Rendering is handled by Gazebo."""
