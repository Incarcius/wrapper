# Setup Guide: Jetson Orin Nano + Livox Mid-70

One-time setup, starting from a new Jetson Orin Nano still in its box. Follow each step in order
and paste commands into the terminal where shown.

When you're done, run the demo using [demo_runbook.md](demo_runbook.md).

| Phase | What you'll have at the end |
|---|---|
| [1. Flash JetPack](#phase-1-flash-the-operating-system-jetpack) | Jetson boots into Ubuntu 22.04 |
| [2. Network the LiDAR](#phase-2-connect-and-network-the-lidar) | Jetson can ping the Mid-70 |
| [3. Install ROS 2 and the driver](#phase-3-install-ros-2-humble-and-the-livox-driver) | Driver builds |
| [4. Verify the LiDAR stream](#phase-4-verify-the-lidar-stream) | Point clouds arriving at ~10 Hz |
| [5. Install and run the wrapper](#phase-5-install-and-run-the-wrapper) | Pipeline processing live frames |

---

## Phase 1: Flash the operating system (JetPack)

A Jetson Orin Nano on recent firmware (e.g. v36.4.7) is ready for JetPack 6.x. These steps use
the SD card image method from a Windows PC.

**You need:** a Windows PC, a good-quality microSD card (64 GB or more) and an SD card reader.

### 1.1 Download the files

1. **Rufus:** download the standalone executable from [rufus.ie](https://rufus.ie/). Use Rufus
   rather than BalenaEtcher, which can crash because of Squirrel conflicts.
2. **OS image:** download the **JetPack 6.x SD Card Image** from the NVIDIA Jetson archive.
   Don't unzip it.
   > ⚠️ Do **not** use JetPack 7.x. NVIDIA removed direct SD card flashing in version 7.

### 1.2 Flash the SD card

1. Open Rufus and select your microSD card under **Device**.
2. Under **Boot selection**, click **SELECT** and choose the JetPack 6.x `.zip` file.
3. Click **START**. Wiping and flashing takes 15–30 minutes.
4. Put the microSD card in the Jetson.
5. Connect a DisplayPort monitor and a USB keyboard and mouse, then plug in the Jetson's power.
6. Complete the Ubuntu setup on screen: username, password and Wi-Fi.

---

## Phase 2: Connect and network the LiDAR

The Mid-70 streams its data over Ethernet. The Jetson needs a static IP on the LiDAR's subnet.

1. **Power the LiDAR:** connect the Mid-70 to its power adapter.
2. **Connect the cable:** run an Ethernet cable from the Mid-70's converter box to the Jetson's
   Ethernet port.
3. **Set a static IP on the Jetson:**
   - Click the **network icon** (top right) → **Wired Settings** → the **gear icon** next to *Wired*.
   - On the **IPv4** tab, change the method from *Automatic (DHCP)* to **Manual**.
   - Address `192.168.1.50`, Netmask `255.255.255.0`, Gateway blank.
   - Click **Apply**, then turn the wired connection off and on again.
4. **Test the connection.** Open a terminal (`Ctrl + Alt + T`) and ping the LiDAR:
   ```bash
   ping 192.168.1.1XX
   ```
   A Livox LiDAR's factory IP is normally `192.168.1.1XX`, where **XX** is the last two digits of
   its serial number. For example, a serial ending in …42 means `192.168.1.142`. Replies like
   `64 bytes from … time=1 ms` mean the network is working. Press `Ctrl + C` to stop.

---

## Phase 3: Install ROS 2 Humble and the Livox driver

JetPack 6 is Ubuntu 22.04, which uses **ROS 2 Humble**. Foxy does not support 22.04.

### 3.1 Add the ROS 2 package sources

```bash
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### 3.2 Install ROS 2

```bash
sudo apt update
sudo apt install ros-humble-ros-base python3-argcomplete -y
```

### 3.3 Build the Livox ROS 2 driver

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/Livox-SDK/livox_ros2_driver.git
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build
```

Things to know about the driver:

- **Use `livox_ros2_driver`, not `livox_ros_driver2`.** The names are confusingly similar.
  `livox_ros_driver2` is for Livox's newer SDK 2 devices (HAP, Mid-360) and does not support the
  Mid-70.
- **It builds its own copy of the Livox SDK.** You don't need to install Livox-SDK separately.
  If Livox-SDK was ever installed system-wide on this Jetson, the driver's README says to rename
  `/usr/local/lib/livox_sdk_static.a` before building.
- **No config edit is needed for a single LiDAR.** By default the driver connects to every Livox
  LiDAR it finds. With several LiDARs on the network, pick yours by broadcast code in
  `livox_lidar_config.json`, in the driver's `launch` folder.

**Optional:** source ROS 2 automatically in every new terminal.

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

---

## Phase 4: Verify the LiDAR stream

Check that the driver publishes data before involving the wrapper. This separates hardware and
driver problems from wrapper problems.

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch livox_ros2_driver livox_lidar_launch.py
```

Leave that running. In a **second terminal**, after sourcing the same two files:

```bash
ros2 topic list               # should include /livox/lidar
ros2 topic hz /livox/lidar    # should settle at about 10 Hz
```

If both look right, stop the driver with `Ctrl + C`. If `/livox/lidar` never appears, read the
driver's output in the first terminal. It usually names the problem, such as no LiDAR found or a
network error.

---

## Phase 5: Install and run the wrapper

### 5.1 Copy the wrapper onto the Jetson

Copy the `wrapper` folder into your home folder (`~/wrapper`) with a USB drive, or clone it if
the Jetson can reach the repository:

```bash
git clone https://github.com/Incarcius/wrapper.git ~/wrapper
```

### 5.2 Install the Python packages

```bash
cd ~/wrapper
sudo apt install python3-pip
pip3 install -r requirements.txt
```

If `open3d` fails to install, check `python3 --version`. Prebuilt ARM64 Open3D wheels exist only
for particular Python versions, and JetPack 6 ships Python 3.10.

### 5.3 Install PyTorch (only once a classifier is added)

PyTorch is **not** in `requirements.txt` on purpose. The PyPI build has no CUDA support on a
Jetson, so a model would silently run on the CPU. Install NVIDIA's JetPack 6 wheel instead,
following the [NVIDIA PyTorch for Jetson forum post](https://forums.developer.nvidia.com/t/pytorch-for-jetson).
Then confirm it can see the GPU:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"    # must print True
```

### 5.4 Run it

```bash
cd ~/wrapper
chmod +x scripts/start_demo.sh    # once; USB copies lose the executable bit
./scripts/start_demo.sh
```

**What success looks like:**

1. `[INFO]` lines confirming dependencies, ROS 2 sourcing and the driver launch.
2. `Pipeline running. Ctrl+C to stop.`
3. About every 3 seconds (every 30 frames at 10 Hz), a timing line such as:
   ```
   2026-09-22 14:03:11,482 INFO livox_wrapper.pipeline: frame=30 roi=0.4ms outlier=11.8ms norm=0.3ms down=24.9ms total=37.4ms
   ```

Nothing else is printed per frame until a classifier is wired in. If no timing lines appear, see
the troubleshooting table in [demo_runbook.md](demo_runbook.md#troubleshooting).

### 5.5 Optional: maximum performance

Put the Jetson in its highest power mode, using the power menu in the top bar or
`sudo nvpmodel -q` to see the current mode. Then lock the clocks at maximum:

```bash
sudo jetson_clocks
```

---

## Classroom testing vs. outdoor flight

`config/pipeline_config.yaml` uses a **±10 m crop box**, sized for indoor proof-of-work testing.
Before outdoor or city-scale flights:

- **Widen the box, keeping the sensor's range in mind.** The Mid-70 reaches about 90 m on
  10%-reflective targets, so a box much beyond ±90 m gains little.
- **Wider doesn't mean more points.** The Mid-70 emits a fixed ~100,000 points per second. A wider
  box spreads the same points over more space, so each frame gets *sparser*, not larger. At long
  range the risk is objects too sparse to classify, not the Jetson being overwhelmed.
- **Account for motion.** Each frame is gathered over 100 ms. A moving drone smears the cloud
  unless points are motion-compensated. The Mid-70 has no built-in IMU, so this needs IMU data
  from the flight controller. See [roadmap.md](roadmap.md).
