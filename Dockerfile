FROM osrf/ros:noetic-desktop-full

WORKDIR /home/catkin_ws/src/rl_uav

COPY . .
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install git python3-pip ros-noetic-geographic-msgs tightvncserver dbus-x11 xfonts-base -y \
    && pip install -r requirements.txt \
    && cd .. \
    && git clone https://github.com/tu-darmstadt-ros-pkg/hector_quadrotor  \
    && git clone https://github.com/tu-darmstadt-ros-pkg/hector_localization  \
    && git clone https://github.com/tu-darmstadt-ros-pkg/hector_gazebo  \
    && git clone https://github.com/tu-darmstadt-ros-pkg/hector_models  \
    && git clone https://github.com/ros-simulation/gazebo_ros_pkgs
