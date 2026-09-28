#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HOST_ROOT=$(dirname "$SCRIPT_DIR")

DRONE_ID=${DRONE_ID:-1}
MAP_DIR=${MAP_DIR:-maps/mph}

# Initial pose: meters in x-forward/y-left; yaw is CCW-positive in degrees.
INITIAL_X=${INITIAL_X:-8.3}
INITIAL_Y=${INITIAL_Y:-8.3}
INITIAL_YAW=${INITIAL_YAW:-0}
# E.g., set x to 9.0: INITIAL_X=${INITIAL_X:-9.0}
# E.g., set yaw to -45: INITIAL_YAW=${INITIAL_YAW:--45}

LIDAR_TOPIC=${LIDAR_TOPIC:-/livox/lidar}
POSE_TOPIC=${POSE_TOPIC:-/agent00${DRONE_ID}/oneshot_localization_result}
DOCKER_IMAGE=${DOCKER_IMAGE:-localisation:fg2}
PYTHON_SCRIPT=${PYTHON_SCRIPT:-mph_ros_test_nonlearn.py}

sudo docker run -it --rm \
  --runtime nvidia \
  --network host \
  --ipc host \
  -v "${HOST_ROOT}:/home/emnavi/X280/src/localization" \
  -e TORCH_HOME=/home/emnavi/X280/src/localization/CVL/torch_cache \
  -w /home/emnavi/X280/src/localization/CVL \
  "${DOCKER_IMAGE}" \
  python "${PYTHON_SCRIPT}" \
    --drone_id "${DRONE_ID}" \
    --map_dir "${MAP_DIR}" \
    --initial_pose "${INITIAL_X}" "${INITIAL_Y}" "${INITIAL_YAW}" \
    --lidar_topic "${LIDAR_TOPIC}" \
    --pose_topic "${POSE_TOPIC}"
