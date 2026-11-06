# Demo Runbook

Day-of checklist for running the live demo on the Jetson Orin Nano. It assumes the one-time
[setup guide](setup_guide.md) is complete.

---

## Before the day

- [ ] The [setup guide](setup_guide.md) is complete, including **Phase 4**: `ros2 topic hz /livox/lidar`
      has shown ~10 Hz on *this* Jetson at least once.
- [ ] The crop box in `config/pipeline_config.yaml` suits the room. The default is ±10 m
      horizontally and −2 m to +3 m vertically, in the sensor frame.
- [ ] You've done a full dry run with `./scripts/start_demo.sh` and seen timing lines.

---

## On the day

### 1. Hardware

1. Power the Mid-70 from its adapter.
2. Connect the Ethernet cable from the Mid-70's converter box to the Jetson.
3. Power the Jetson and log in.

### 2. Check the network

```bash
ping 192.168.1.1XX      # XX = last two digits of the LiDAR's serial number
```

Replies mean the network is fine. Press `Ctrl + C` to stop.

### 3. Optional: maximum performance

```bash
sudo jetson_clocks
```

### 4. Start the demo

```bash
cd ~/wrapper
./scripts/start_demo.sh
```

### 5. What you should see

```
=================================================
 Starting Livox Mid-70 preprocessing pipeline
=================================================
[INFO] Checking Python dependencies...
[INFO] All Python dependencies found.
[INFO] Sourcing ROS2 humble...
[INFO] Sourcing local workspace at /home/<you>/ros2_ws...
[INFO] Launching livox_ros2_driver...
[INFO] Driver started in background (PID: 12345).
[INFO] Waiting 3 seconds for LiDAR initialization...
[INFO] Starting Python processing pipeline...
... INFO livox_wrapper.cli: Pipeline running. Ctrl+C to stop.
... INFO livox_wrapper.pipeline: frame=30 roi=0.4ms outlier=11.8ms norm=0.3ms down=24.9ms total=37.4ms
... INFO livox_wrapper.pipeline: frame=60 ...
```

A timing line appears about every 3 seconds (every 30 frames at 10 Hz). **Between timing lines
the terminal is quiet, and that's normal:** no classifier is wired in yet, so nothing is printed
per frame. Your numbers will differ from the example. `total` should stay under ~100 ms to keep
up with the sensor.

### 6. Stop

Press `Ctrl + C`. Then check that the driver stopped too, because the script has no `trap`, so
its cleanup step isn't guaranteed to run
([code review N5](code_review.md#-nice-to-have)):

```bash
pgrep -af livox                  # should print nothing
pkill -f livox_ros2_driver       # only if something is still running
```

A driver left running can keep hold of the LiDAR's network ports, and then the next start may
fail.

---

## Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `/bin/bash^M: bad interpreter` | The script has Windows (CRLF) line endings from a copy made on Windows | `sed -i 's/\r$//' scripts/start_demo.sh`. Fresh checkouts are protected by `.gitattributes`. |
| `Permission denied` when starting the script | The executable bit was lost when copying by USB | `chmod +x scripts/start_demo.sh` |
| `[ERROR] ROS2 setup file not found` | ROS 2 isn't installed, or it's a different distro | `ls /opt/ros/` shows what's installed. Set `ROS2_DISTRO` at the top of the script to match. |
| `[ERROR] Missing required Python packages` | Dependencies aren't installed for `python3` | `pip3 install -r requirements.txt` |
| `Pipeline running` appears, but **no timing lines ever appear** | Either the driver isn't publishing, or frames are being dropped as stale | See [No timing lines](#no-timing-lines) below |
| `Exception in thread …` / `RuntimeError: ROS2 ingestion mode requires rclpy`, then silence | ROS 2 isn't sourced for this Python, or `sensor_msgs_py` is missing | The error happens in the background thread, so the main loop keeps waiting silently ([code review I2](code_review.md#i2--an-exception-in-the-ingestion-thread-kills-ingestion-silently-the-main-loop-spins-forever)). Press `Ctrl + C`, check `source /opt/ros/humble/setup.bash` and `python3 -c "import rclpy, sensor_msgs_py"`, then try again. |
| `RuntimeError: outlier_removal.method requires open3d` | Open3D isn't installed | `pip3 install open3d`, or set `outlier_removal.method: none` and `downsampling.method: random` to run without it |
| `ping` gets no reply | Static IP not applied, or wrong LiDAR IP | Re-check [setup guide Phase 2](setup_guide.md#phase-2-connect-and-network-the-lidar). Toggle the wired connection off and on. |
| Timing lines appear, but clouds are sometimes empty or truncated | Large ROS 2 messages are losing UDP fragments | `sudo sysctl -w net.core.rmem_max=2147483647` ([code review I3](code_review.md#i3--default-qos-reliable-depth-10-works-against-the-drop-stale-design)) |

### No timing lines

Work through these in order:

**1. Is the driver publishing?** In a second terminal:

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 topic hz /livox/lidar
```

**No output?** The driver isn't publishing. The demo script discards the driver's output
([code review C4](code_review.md#c4--start_demosh-hides-the-drivers-output)), so stop everything
and run the driver by hand to see its errors:

```bash
ros2 launch livox_ros2_driver livox_lidar_launch.py
```

**2. Is `hz` about 10 Hz, but still no timing lines?** Frames are probably being dropped as
"stale". The wrapper compares each frame's sensor timestamp with the Jetson's clock, and if the
two clocks disagree, it drops every frame
([code review C3](code_review.md#c3--frame-staleness-is-checked-against-the-wrong-clock-it-can-silently-drop-100-of-frames)).
As a demo-day workaround, turn the check off in `config/pipeline_config.yaml`:

```yaml
ingestion:
  drop_stale_after_sec: 1000000   # effectively disables the stale-frame check
```

**3. Still nothing?** Check the crop box. If `roi_filter` removes every point, the pipeline keeps
running and logs `Empty point cloud reached FPS stage; returning zeros.` As a test, set
`roi_filter.mode: none`.
