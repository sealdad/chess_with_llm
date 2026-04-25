#!/bin/bash
source /opt/ros/noetic/setup.bash
source /interbotix_ws/devel/setup.bash
export PYTHONPATH="/interbotix_ws/devel/lib/python3/dist-packages:/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/src:${PYTHONPATH}"
exec python3 -m uvicorn main:app --host 0.0.0.0 --port 8002
