#!/usr/bin/env python
"""This module contains the basic Navigation environment class."""
import math
from typing import Tuple, Optional, Any, Dict, List

import numpy as np
import rospy
import tf
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Twist
from gym import Env
from gym.core import RenderFrame
from gym.spaces import Discrete, Box
from hector_uav_msgs.srv import EnableMotors
from rospy.topics import Publisher
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from std_srvs.srv import Empty

from rl_uav.data_classes.euler import Euler
from rl_uav.enums.action import Action
from rl_uav.enums.track import Track


class Navigation(Env):
    """The basic Navigation environment."""
    _WAIT_TIME = rospy.Duration.from_sec(8)
    _STEP_DURATION = rospy.Duration.from_sec(0.4)

    _COLLISION_REWARD = -200.0
    _FORWARD_REWARD = +5.0
    _YAW_REWARD = -0.5

    _FLYING_ALTITUDE = 1.5  # m
    _COLLISION_THRESHOLD = 0.4  # m
    _INIT_ALTITUDE = 4  # m

    _VELOCITY_STANDARD_DEVIATION = 0.01

    _MEASUREMENTS = (180, 360, 540, 720, 900)

    _N_OBSERVATIONS = len(_MEASUREMENTS)

    _spawn_area: Track
    _ranges: Optional[np.ndarray]
    _ranges_range: Tuple[float, float]
    _vel_pub: Publisher
    _pose_pub: Publisher

    def __init__(self,
                 render_mode: Optional[str] = None,
                 track_id: int = 1) -> None:
        if (render_mode is not None
                and render_mode not in self.metadata['render_modes']):
            raise ValueError(f'Mode {render_mode} is not supported')

        self.render_mode = render_mode  # type: ignore
        self._track = Track(track_id)

        self._ranges = None
        self._ranges_range = (-math.inf, math.inf)

        self._vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self._pose_pub = rospy.Publisher('command/pose',
                                         PoseStamped,
                                         queue_size=10)

        rospy.Subscriber('scan', LaserScan, self._laser_scan_callback)

        while not Navigation._enable_motors():
            pass

        while self._ranges is None:
            pass

        self.action_space = Discrete(len(Action))

        high = np.array(self._N_OBSERVATIONS * [self._ranges_range[0]],
                        dtype=np.float64)
        low = np.array(self._N_OBSERVATIONS * [self._ranges_range[1]],
                       dtype=np.float64)
        self.observation_space = Box(low=low,
                                     high=high,
                                     shape=(self._N_OBSERVATIONS,),
                                     dtype=np.float64)

    def _perform_action(self, action: int) -> None:
        vel_msg = Twist()

        action_enum = Action(action)

        vel_msg.linear.x = (
                action_enum.linear_velocity
                + self.np_random.normal(0, self._VELOCITY_STANDARD_DEVIATION))
        vel_msg.angular.z = (
                action_enum.angular_velocity
                + self.np_random.normal(0, self._VELOCITY_STANDARD_DEVIATION))

        self._vel_pub.publish(vel_msg)

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

        self._pose_pub.publish(pose_stamped)

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

    def reset(self,
              *,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None
              ) -> np.ndarray | Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        Navigation._reset_world()
        rospy.sleep(self._WAIT_TIME)

        area = self.np_random.choice(self._track.spawn_area)
        x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
        y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
        yaw = self.np_random.uniform(-math.pi, math.pi)

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

        return (self._ranges, {}) if return_info else self._ranges

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action {action} ({type(action)})')

        self._perform_action(action)
        rospy.sleep(self._STEP_DURATION)

        if self._ranges is None:
            raise ValueError

        terminated = self._collision_occurred()
        truncated = False

        if terminated:
            reward = self._COLLISION_REWARD
        elif action == Action.FORWARD.value:
            reward = self._FORWARD_REWARD
        else:
            reward = self._YAW_REWARD

        return self._ranges, reward, terminated, truncated, {}

    def render(self,
               mode: str = 'human'
               ) -> Optional[List[RenderFrame]]:
        """Rendering is handled by Gazebo."""
