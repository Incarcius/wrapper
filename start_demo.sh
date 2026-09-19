#!/bin/bash
# ==============================================================================
# start_demo.sh
# 
# Automated startup script for the Livox LiDAR point cloud pipeline on Jetson Nano.
# This script sources ROS2, launches the Livox driver, and starts the ML pipeline.
# ==============================================================================

# --- CONFIGURATION ---
# Edit these paths to match your actual Jetson Nano environment!
ROS2_DISTRO="foxy"                     # Usually 'foxy' or 'humble' on Jetson
ROS2_WORKSPACE="$HOME/ros2_ws"         # Path to your ROS2 workspace containing livox_ros_driver
PYTHON_EXEC="python3"                  # 'python3' or path to a venv (e.g., 'venv/bin/python')
# ---------------------

echo "================================================="
echo " Starting Livox Mid-70 Wrapper & DGCNN Pipeline  "
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
    echo "          (Ignore this if the livox_ros_driver is installed system-wide)"
fi

# 3. Launch the Livox ROS2 Driver in the background
#    (Modify the launch command below if your version of the driver uses a different launch file)
echo "[INFO] Launching livox_ros_driver..."
ros2 launch livox_ros_driver livox_lidar_launch.py > /dev/null 2>&1 &
DRIVER_PID=$!
echo "[INFO] Driver started in background (PID: $DRIVER_PID)."

# Give the LiDAR hardware a few seconds to spin up and connect
echo "[INFO] Waiting 3 seconds for LiDAR initialization..."
sleep 3

# 4. Start the Python Processing Pipeline
echo "[INFO] Starting Python processing pipeline..."
# Change to the directory where this bash script is located
cd "$(dirname "$0")" || exit

# Run the pipeline (assumes src/run_pipeline.py and config/pipeline_config.yaml exist here)
$PYTHON_EXEC src/run_pipeline.py --config config/pipeline_config.yaml

# 5. Cleanup when Python script exits (e.g., when you press Ctrl+C)
echo "================================================="
echo "[INFO] Stopping Livox driver (PID: $DRIVER_PID)..."
kill $DRIVER_PID
echo "Pipeline gracefully stopped."
