# 📖 Complete Beginner's Guide: Jetson Orin Nano + Livox Mid-70 

This guide assumes you are starting from absolute scratch with a brand new Jetson Orin Nano in a box. Follow every step exactly, copying and pasting commands into the terminal where instructed.

---

## Phase 1: Flashing the Operating System (JetPack)

Because your Jetson Orin Nano shipped with brand new firmware (v36.4.7), it is natively ready for JetPack 6.x. We will use the **SD Card Image** method using Windows.

### What you need:
- A host computer (Windows).
- A high-quality MicroSD card (at least 64GB).
- An SD card reader.

### Step 1.1: Download the Files
1. **Download Rufus:** Go to [rufus.ie](https://rufus.ie/) and download the standalone executable. (We use Rufus because BalenaEtcher can crash due to Squirrel conflicts).
2. **Download the OS Image:** Go to the NVIDIA Jetson Archive and download the **JetPack 6.x SD Card Image** (Do not unzip it).
   - ⚠️ *Note: Do NOT use JetPack 7.x! NVIDIA removed direct SD card flashing in version 7.*

### Step 1.2: Flash the SD Card
1. Open Rufus.
2. Select your MicroSD card under "Device".
3. Under "Boot selection", click **SELECT** and choose the **JetPack 6.x** `.zip` file.
4. Click **START** and let it wipe and flash the drive (this will take 15-30 mins).
5. When finished, take the MicroSD card out of the laptop and insert it into the Jetson.
6. Connect a DisplayPort cable to a monitor, plug in a USB keyboard/mouse, and plug in the Jetson's power.
7. Follow the on-screen Ubuntu setup (choose a username, password, and connect to Wi-Fi).

---

## Phase 2: Hardware Connections & Networking

The Livox LiDAR sends massive amounts of data over Ethernet. We must tell the Jetson exactly where to look for it.

### How to do it:
1. **Power the LiDAR:** Connect the Livox Mid-70 to its power adapter and plug it into the wall.
2. **Connect the Cable:** Take an Ethernet cable and plug one end into the Mid-70's converter box, and the other end directly into the Jetson Orin Nano's Ethernet port.
3. **Set a Static IP:**
   - On the Jetson's desktop, click the **Network Icon** in the top right corner.
   - Click **Wired Settings**.
   - Next to the "Wired" connection, click the **Gear Icon** (⚙️).
   - Go to the **IPv4** tab at the top.
   - Change the IPv4 Method from "Automatic (DHCP)" to **Manual**.
   - Under "Addresses", fill in the following:
     - **Address:** `192.168.1.50`
     - **Netmask:** `255.255.255.0`
     - **Gateway:** Leave blank
   - Click **Apply**. Turn the connection OFF and then ON again using the toggle switch to apply it.
4. **Test the Connection:**
   - Open a terminal by pressing `Ctrl + Alt + T` on your keyboard.
   - Type this command and press Enter: `ping 192.168.1.100` (Note: the LiDAR might be 192.168.1.something else depending on its serial number, but 1.100 is common).
   - If you see `64 bytes from... time=1ms`, it is working! Press `Ctrl + C` to stop pinging.

---

## Phase 3: Installing ROS2 and Dependencies

We need ROS2 (Robot Operating System) to efficiently talk to the LiDAR. Since we are likely on Ubuntu 22.04 (JetPack 6), we will install **ROS2 Humble**.

### How to do it (Copy and paste line by line):

1. **Open a new terminal (`Ctrl + Alt + T`).**
2. **Setup ROS2 Sources:**
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
3. **Install ROS2:**
   ```bash
   sudo apt update
   sudo apt install ros-humble-ros-base python3-argcomplete -y
   ```
4. **Install the Livox ROS2 Driver:**
   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws/src
   git clone https://github.com/Livox-SDK/livox_ros2_driver.git
   cd ~/ros2_ws
   source /opt/ros/humble/setup.bash
   colcon build
   ```
5. **Transfer the Wrapper:** 
   - Take the `wrapper` folder you have on your PC and put it onto a USB thumb drive.
   - Plug the USB into the Jetson, copy the `wrapper` folder, and paste it into the Jetson's **Home** folder.
6. **Install Python Packages:**
   ```bash
   cd ~/wrapper
   sudo apt install python3-pip
   pip3 install -r requirements.txt
   ```
   *(Note: PyTorch for Jetson requires a specific file. If standard `pip3 install torch` fails, go to the [NVIDIA Jetson PyTorch forum post](https://forums.developer.nvidia.com/t/pytorch-for-jetson) and download the wheel file for JetPack 6, then install it).*

---

## Phase 4: Testing the Wrapper (Your PoW)

### ⚠️ IMPORTANT NOTE FOR THE FUTURE (City vs Classroom)
Currently, `pipeline_config.yaml` is heavily restricted to look at a **10-meter box** (a classroom) to save processing power. When this project matures and you mount it on a drone to scan **cities**, you MUST open `wrapper/config/pipeline_config.yaml` and increase the `x_max/y_max` to `100.0` meters, AND increase the `voxel_size_m` so the Jetson doesn't crash from processing millions of points.

### How to run the demo today:

1. **Open a terminal (`Ctrl + Alt + T`).**
2. **Navigate to the wrapper:**
   ```bash
   cd ~/wrapper
   ```
3. **Make the script executable (you only do this once):**
   ```bash
   chmod +x start_demo.sh
   ```
4. **Run the demo:**
   ```bash
   # If you installed ROS2 Humble, open start_demo.sh in a text editor
   # and make sure line 11 says: ROS2_DISTRO="humble"
   
   ./start_demo.sh
   ```

**Success looks like this:**
The terminal will start printing green logs. Within 3 seconds, you will see timing logs (e.g., `roi=1.2ms outlier=5.1ms`) and the script will output a matrix shaped `(1024, 3)`. If you see this, your LiDAR is talking to the Jetson, the wrapper is cleaning the data, and your Deep Learning Pipeline is officially ready for a model!
