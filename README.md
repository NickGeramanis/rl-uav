# Reinforcement Learning for Autonomous Unmanned Aerial Vehicles - Undergraduate Thesis

Implementation of different reinforcement learning algorithms to solve the
navigation problem of an unmanned aerial vehicle (UAV)
using [ROS](https://www.ros.org/)/[Gazebo](http://gazebosim.org/) and Python.

## Table of Contents

- [Description](#description)
    - [Abstract](#abstract)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
- [Usage](#usage)
- [Status](#status)
- [License](#license)
- [Authors](#authors)

## Description

`navigation_env.py`: The goal of this environment is to navigate a robot on a
track without crashing into the walls. Initially, the robot is placed randomly
into the track but at a safe distance from the walls. The state-space consists
of 5 range measurements. The action-space consist of 3 actions (move_forward,
rotate_left, rotate_right). Furthermore, both actions and states have additive
white Gaussian noise. The robot is rewarded +5 for moving forward and -0.5 for
rotating. If the robot crashes into the wall it is penalized with -200.

There are 3 available worlds/tracks:

Track1:

![Track1](/images/track1.png)

Track2:

![Track2](/images/track2.png)

Track3:

![Track3](/images/track3.png)

This project has been structured based
on [OpenAI Gym](https://github.com/openai/gym) framework.

### Abstract

Reinforcement learning is an area of machine learning concerned with how
autonomous agents learn to behave in unknown environments through
trial-and-error. The goal of a reinforcement learning agent is to learn a
sequential decision policy that maximizes the notion of cumulative reward
through continuous interaction with the unknown environment. A challenging
problem in robotics is the autonomous navigation of an Unmanned Aerial
Vehicle (UAV) in worlds with no available map. This ability is critical in many
applications, such as search and rescue operations or the mapping of
geographical areas. In this thesis, we present a map-less approach for the
autonomous, safe navigation of a UAV in unknown environments using
reinforcement learning. Specifically, we implemented two popular algorithms,
SARSA(λ) and Least-Squares Policy Iteration (LSPI), and combined them with tile
coding, a parametric, linear approximation architecture for value function in
order to deal with the 5- or 3-dimensional continuous state space defined by
the measurements of the UAV distance sensors. The final policy of each
algorithm, learned over only 500 episodes, was tested in unknown environments
more complex than the one used for training in order to evaluate the behavior
of each policy. Results show that SARSA(λ) was able to learn a near-optimal
policy that performed adequately even in unknown situations, leading the UAV
along paths free-of-collisions with obstacles. LSPI's policy required less
learning time and its performance was promising, but not as effective, as it
occasionally leads to collisions in unknown situations. The whole project was
implemented using the Robot Operating System (ROS) framework and the Gazebo
robot simulation environment.

Supervisor: Associate Professor Michail G. Lagoudakis

https://doi.org/10.26233/heallink.tuc.87066

## Getting Started

### Prerequisites

This package has only been tested in ROS Kinetic (Ubuntu 16.04) - Python 2.7.

Required ROS packages:

- hector_quadrotor
- hector_slam
- hector_localization
- hector_gazebo
- hector_models
- gazebo-ros-pkgs

It is recommended that you build the above packages from source (clone the
corresponding git repositories)

Required Python libraries:

- NumPy

### Installation

Clone the repository into your catkin workspace:

```bash
cd ~/catkin_ws/src
clone https://github.com/NickGeramanis/undergrad-thesis-rl-uav.git
```

Build your catkin workspace:

```bash
cd ~/catkin_ws
catkin_make
```

Do not forget to source the new `setup.*sh` file:

```bash
cd ~/catkin_ws
source devel/setup.bash
```

## Usage

In order to launch a new world you must start the `train.launch` file:

```bash
roslaunch uav_rl train.launch world:=track1 gui:=true
```

After the world has started, run the `train_uav` node (`train_uav.py`) to begin
the training process and test different algorithms:

```bash
rosrun uav_rl train_uav.py
```

## Status

Under maintenance.

## License

Distributed under the GPL-3.0 License. See `LICENSE` for more information.

## Authors

[Nick Geramanis](https://www.linkedin.com/in/nikolaos-geramanis)

