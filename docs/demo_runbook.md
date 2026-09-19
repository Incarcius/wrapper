# Live Demo Runbook: Livox Mid-70 + Jetson Nano

Follow this step-by-step checklist tomorrow when you have the hardware in front of you to ensure a smooth Proof-of-Concept (POC) demonstration.

---

## 1. Hardware Connections
1. **Power Up:** Turn on the Jetson Nano and power up the Livox Mid-70 using its power adapter.
2. **Network Connection:** Connect the Livox Mid-70 to the Jetson Nano via an Ethernet cable. 
3. **Configure IP Address:** 
   - By default, the Livox LiDAR expects the host machine to be on the `192.168.1.x` subnet. 
   - On the Jetson Nano, go to Network Settings -> Wired Connection and set a **Manual/Static IPv4 Address** (e.g., Address: `192.168.1.50`, Netmask: `255.255.255.0`). 
   - *Tip: If you can successfully `ping 192.168.1.1xx` (the LiDAR's IP), your hardware is connected correctly.*

## 2. Transfer the Code
1. Copy this entire `wrapper/` folder from your PC onto the Jetson Nano (e.g., via a USB flash drive or SSH).
2. Place it somewhere accessible, like your Desktop or Home folder (`~/wrapper/`).

## 3. Install Python Dependencies
Open a terminal on the Jetson Nano and navigate to the wrapper folder:
```bash
cd ~/wrapper
```
Install the required ML and math packages:
```bash
pip3 install -r requirements.txt
```
*(Note: If you run into issues installing `open3d` or `torch` via pip, you may need to download the official Nvidia Jetson `.whl` files from the Nvidia forums).*

## 4. Prepare the Startup Script
While still in the `wrapper/` folder, make sure the startup script has permission to execute:
```bash
chmod +x start_demo.sh
```
*(Optional: Open `start_demo.sh` in a text editor and double-check that the `ROS2_DISTRO` and `ROS2_WORKSPACE` variables match your Jetson's exact setup).*

## 5. Verify the ROS2 Driver (One-Time Check)
Before running the automated script, ensure the Livox C++ driver is ready:
1. Open a terminal and source your workspace: `source ~/ros2_ws/install/setup.bash` (or wherever your driver is built).
2. If you haven't built the driver yet on this Jetson, clone the official `livox_ros_driver` into `~/ros2_ws/src`, run `colcon build`, and source it.

## 6. SHOWTIME: Run the Demo
Once the LiDAR is spinning and your Jetson is ready, simply run:
```bash
./start_demo.sh
```

**What to expect on the screen:**
1. You will see green `[INFO]` tags confirming Python dependencies are found.
2. The script will automatically launch the ROS2 hardware driver in the background.
3. 3 seconds later, the Python pipeline will boot up.
4. You should immediately start seeing the DGCNN matrices `(1, 3, 1024)` being printed or logged to the console, demonstrating real-time ingestion, cleaning, downsampling, and formatting! 

When you are done, simply press `Ctrl+C` and the script will safely shut down the LiDAR driver. Good luck!
