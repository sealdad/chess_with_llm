# Robot Service Dockerfile
# Based on ROS1 Noetic with Interbotix SDK

FROM ros:noetic-ros-base

# Set working directory
WORKDIR /app

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install ROS dependencies for Interbotix
RUN apt-get update && apt-get install -y \
    ros-noetic-ros-control \
    ros-noetic-ros-controllers \
    ros-noetic-gazebo-ros \
    ros-noetic-gazebo-ros-control \
    ros-noetic-joint-state-publisher \
    ros-noetic-joint-state-publisher-gui \
    ros-noetic-robot-state-publisher \
    ros-noetic-xacro \
    ros-noetic-tf2-tools \
    ros-noetic-tf2-geometry-msgs \
    ros-noetic-dynamixel-sdk \
    ros-noetic-moveit \
    ros-noetic-moveit-ros-planning \
    ros-noetic-moveit-ros-move-group \
    ros-noetic-apriltag-ros \
    ros-noetic-realsense2-camera \
    ros-noetic-realsense2-description \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create catkin workspace
RUN mkdir -p /interbotix_ws/src

# Clone Interbotix ROS packages
WORKDIR /interbotix_ws/src
RUN git clone --depth 1 -b noetic https://github.com/Interbotix/interbotix_ros_core.git && \
    git clone --depth 1 -b noetic https://github.com/Interbotix/interbotix_ros_manipulators.git && \
    git clone --depth 1 -b noetic https://github.com/Interbotix/interbotix_ros_toolboxes.git

# Remove CATKIN_IGNORE files to enable required packages
RUN rm -f /interbotix_ws/src/interbotix_ros_core/interbotix_ros_xseries/CATKIN_IGNORE && \
    rm -f /interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/CATKIN_IGNORE && \
    rm -f /interbotix_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/CATKIN_IGNORE && \
    touch /interbotix_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/CATKIN_IGNORE

# Install dependencies via rosdep
WORKDIR /interbotix_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y || true"

# Build the workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    catkin_make -DCMAKE_BUILD_TYPE=Release"


# Install Python dependencies for interbotix_xs_modules
RUN pip3 install modern_robotics transforms3d

# Install interbotix_xs_modules Python package
RUN cd /interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules && \
    pip3 install -e .

# Install interbotix_common_modules if it has setup.py
RUN cd /interbotix_ws/src/interbotix_ros_toolboxes/interbotix_common_toolbox/interbotix_common_modules && \
    pip3 install -e . || true

# Set up environment
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    echo "source /interbotix_ws/devel/setup.bash" >> ~/.bashrc

# Set PYTHONPATH to include all required paths
ENV PYTHONPATH="/interbotix_ws/devel/lib/python3/dist-packages:/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/src:/opt/ros/noetic/lib/python3/dist-packages:${PYTHONPATH}"

# Set ROS environment variables
ENV ROS_PACKAGE_PATH="/interbotix_ws/src:/opt/ros/noetic/share"
ENV LD_LIBRARY_PATH="/interbotix_ws/devel/lib:/opt/ros/noetic/lib:${LD_LIBRARY_PATH}"
ENV CMAKE_PREFIX_PATH="/interbotix_ws/devel:/opt/ros/noetic"

# Back to app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY services/robot/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy service code
COPY services/robot/main.py .
COPY services/robot/test_robot.py .
COPY services/robot/start.sh .
RUN chmod +x /app/start.sh

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run service with ROS and Interbotix environment
CMD ["/bin/bash", "/app/start.sh"]
