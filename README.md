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
        - [With GUI](#with-gui)
- [Usage](#usage)
- [Status](#status)
- [License](#license)
- [Authors](#authors)

## Description

`navigation_env.py`: The goal of this environment is to navigate an UAV on a
track without crashing into the walls. Initially, the UAV is placed randomly
into the track but at a safe distance from the walls. The state space consists
of 5 range measurements. The action space consists of 3 actions (move_forward,
rotate_left, rotate_right). Furthermore, both actions and states have additive
white Gaussian noise. The UAV is rewarded +5 for moving forward and -0.5 for
rotating. If the UAV crashes into the wall it is penalized with -200.

There are 3 available worlds/tracks:

Track1:

![Track1](/images/track1.png)

Track2:

![Track2](/images/track2.png)

Track3:

![Track3](/images/track3.png)

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

This package is provided as a docker image.

### Installation

Build the docker image:

```bash
docker build -t rl-uav .
```

Run the container:

```bash
docker run --name rl-uav -it rl-uav
```

Due to XmlRpcServer querying all possible file descriptors, it may be required to lower the corresponding limit depending on your system:

```bash
docker run --ulimit nofile=1024:524288 --name rl-uav -it rl-uav
```

Build the ros package:

```bash
source /opt/ros/noetic/setup.bash
cd /home/catkin_ws
catkin_make
```

Due to the high resource intensity of the catkin_make, it may be required to run only one job at a time.

```bash
catkin_make -j1
```

#### With GUI

The above image also contains a vnc server in order to display the gazebo simulation.

First create a docker network:

```bash
docker network create ros
```

And then run the container as following:

```bash
docker run --ulimit nofile=1024:524288 --name rl-uav --net=ros --env="DISPLAY=novnc:0.0" --env="RESOLUTION=1920x1080" --env="USER=root" -it rl-uav
```

Start the vnc server with the following command:

```bash
vncserver -geometry $RESOLUTION
```

Run the noVNC client:

```bash
docker run -d --rm --net=ros --env="DISPLAY_WIDTH=1920" --env="DISPLAY_HEIGHT=1800" --env="RUN_XTERM=no" --name=novnc -p=8080:8080 theasp/novnc:latest
```

Connect to novnc using the following url: http://localhost:8080/vnc.html.

## Usage

In order to launch a new world you must start the `train.launch` file.
You can select the desired track by changing the `world` parameter accordingly.
Moreover, you can choose whether to display the GUI with the `gui` parameter.

```bash
source /home/catkin_ws/devel/setup.bash
roslaunch rl_uav train.launch world:=track1 gui:=true
```

After the world has started, run the `train_uav` node (`train_uav.py`) to begin
the training process and test different algorithms:

```bash
source /home/catkin_ws/devel/setup.bash
rosrun rl_uav train_uav.py
```

## Status

Under maintenance.

## License

Distributed under the GPL-3.0 License. See `LICENSE` for more information.

## Authors

[Nick Geramanis](https://www.linkedin.com/in/nikolaos-geramanis)

