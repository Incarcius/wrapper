#!/bin/bash
# ==============================================================================
# start_demo.sh
#
# Startup script for the Livox Mid-70 preprocessing pipeline on a Jetson Orin
# Nano: sources ROS 2, launches the Livox driver, then starts the pipeline.
#
# Run from anywhere:   ./scripts/start_demo.sh
# ==============================================================================

# --- CONFIGURATION ---
# Edit these to match your Jetson environment.
ROS2_DISTRO="${ROS_DISTRO:-humble}"    # JetPack 6 = Ubuntu 22.04 = ROS 2 Humble
ROS2_WORKSPACE="$HOME/ros2_ws"         # workspace containing livox_ros2_driver
DRIVER_PKG="livox_ros2_driver"         # Mid-70 driver (Livox SDK 1); see docs/setup_guide.md
DRIVER_LAUNCH="livox_lidar_launch.py"  # publishes sensor_msgs/PointCloud2, no RViz
PYTHON_EXEC="python3"                  # 'python3' or a venv interpreter, e.g. 'venv/bin/python'
# ---------------------

echo "================================================="
echo " Starting Livox Mid-70 preprocessing pipeline    "
echo "================================================="

# 0. Check Python dependencies
echo "[INFO] Checking Python dependencies..."
$PYTHON_EXEC -c "import numpy, open3d, yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[ERROR] Missing required Python packages (numpy, open3d, pyyaml, etc)."
    echo "        Please install them manually using: pip install -r requirements.txt"
    exit 1
fi
echo "[INFO] All Python dependencies found."

# 1. Source ROS2 environment
if [ -f "/opt/ros/$ROS2_DISTRO/setup.bash" ]; then
    echo "[INFO] Sourcing ROS2 $ROS2_DISTRO..."
    source "/opt/ros/$ROS2_DISTRO/setup.bash"
else
    echo "[ERROR] ROS2 setup file not found at /opt/ros/$ROS2_DISTRO/setup.bash!"
    echo "        Please check the ROS2_DISTRO variable in this script."
    exit 1
fi

# 2. Source the local ROS2 workspace (if it exists)
if [ -f "$ROS2_WORKSPACE/install/setup.bash" ]; then
    echo "[INFO] Sourcing local workspace at $ROS2_WORKSPACE..."
    source "$ROS2_WORKSPACE/install/setup.bash"
else
    echo "[WARNING] Local workspace setup not found at $ROS2_WORKSPACE/install/setup.bash"
    echo "          (Ignore this if $DRIVER_PKG is installed system-wide)"
fi

# 3. Launch the Livox ROS2 Driver in the background
#    NOTE: driver output is discarded. If no frames arrive, run the launch
#    command by hand in another terminal to see its errors (docs/demo_runbook.md).
echo "[INFO] Launching $DRIVER_PKG..."
ros2 launch "$DRIVER_PKG" "$DRIVER_LAUNCH" > /dev/null 2>&1 &
DRIVER_PID=$!
echo "[INFO] Driver started in background (PID: $DRIVER_PID)."

# Give the LiDAR hardware a few seconds to spin up and connect
echo "[INFO] Waiting 3 seconds for LiDAR initialization..."
sleep 3

# 4. Start the Python processing pipeline from the repository root
echo "[INFO] Starting Python processing pipeline..."
cd "$(dirname "$0")/.." || exit 1

# Prepend src/ so the package imports without `pip install`. Keep the existing
# PYTHONPATH: ROS 2's setup.bash puts rclpy on it.
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
$PYTHON_EXEC -m livox_wrapper --config config/pipeline_config.yaml

# 5. Cleanup when Python script exits (e.g., when you press Ctrl+C)
echo "================================================="
echo "[INFO] Stopping Livox driver (PID: $DRIVER_PID)..."
kill $DRIVER_PID
echo "Pipeline gracefully stopped."
