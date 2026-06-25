# ============================================================
# Mavic 2 Pro Drone Controller — Forest Firefighters Project
# Course: Computer Science
# ============================================================
# Aerial firefighting drone that patrols the forest area,
# detects fire and smoke using camera and OpenCV, navigates
# to fires using GPS coordinates, and drops water to extinguish.
# Uses sensor fusion: GPS for localisation, camera for detection,
# IMU and gyro for attitude stabilisation.
# ============================================================

import sys
import traceback
from controller import Robot
import random
import optparse

try:
    import numpy as np
except ImportError:
    sys.exit("Error: 'numpy' module not found. Install with: pip install numpy")
try:
    import cv2
except ImportError:
    sys.exit("Error: 'cv2' module not found. Install with: pip install opencv-python")


def clamp(value, value_min, value_max):
    """Restrict a value to the range [value_min, value_max]."""
    return min(max(value, value_min), value_max)


class Mavic(Robot):
    """
    Autonomous Mavic 2 Pro drone for aerial firefighting.

    Locomotion:
        Four propellers controlled via differential velocity.
        A PID control loop maintains altitude, roll, and pitch stability.
        Yaw and forward pitch are disturbed to produce horizontal movement.

    Navigation:
        1. PATROL mode: fly a square patrol route over the forest
           using GPS waypoints and bearing-based yaw/pitch control.
        2. FIRE mode: supervisor writes fire GPS into customData;
           drone flies directly to fire coordinates using GPS.
        3. Drones are pre-assigned to different fires (one-to-one)
           to prevent both targeting the same fire.

    Perception (Sensor Fusion):
        - Camera captures downward video at each timestep
        - OpenCV converts to HSV and masks smoke-coloured pixels
        - Centroid of detected pixels gives fire image coordinates
        - GPS provides world-coordinate distance and bearing to fire
        - Both sensors are used together: GPS for coarse navigation,
          camera for fine-grained detection confirmation
    """

    # ── PID GAINS ──────────────────────────────────────────────────────────
    K_VERTICAL_THRUST = 68.5  # Base motor speed required to hover (empirical)
    K_VERTICAL_OFFSET = 0.6   # Altitude error offset to keep drone stable at target
    K_VERTICAL_P      = 3.0   # Proportional gain for altitude PID (cubic response)
    K_ROLL_P          = 50.0  # Proportional gain for roll stabilisation
    K_PITCH_P         = 30.0  # Proportional gain for pitch stabilisation

    # ── NAVIGATION LIMITS ──────────────────────────────────────────────────
    MAX_YAW_DISTURBANCE   = 4.0   # Max yaw disturbance (rad/s) for turning
    MAX_PITCH_DISTURBANCE = -4.0  # Max pitch disturbance (negative = forward lean)
    target_precision      = 2.0   # Waypoint reached when within this distance (m)

    def __init__(self):
        Robot.__init__(self)
        self.time_step = int(self.getBasicTimeStep())
        self.water_to_drop = 0  # Quantity of water pending to drop

        # ── SENSORS ────────────────────────────────────────────────────────
        self.camera = self.getDevice("camera")
        self.camera.enable(self.time_step)      # Downward camera for fire detection

        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)          # Roll, pitch, yaw angles

        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)          # World X, Y, Z position

        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)         # Angular acceleration for PID damping

        # ── ACTUATORS ──────────────────────────────────────────────────────
        self.front_left_motor  = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor   = self.getDevice("rear left propeller")
        self.rear_right_motor  = self.getDevice("rear right propeller")

        # Camera gimbal pitched to 1.55 rad (~89°) — points nearly straight down
        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.camera_pitch_motor.setPosition(1.55)

        # Propellers run at infinite position (velocity control mode)
        for motor in [self.front_left_motor, self.front_right_motor,
                      self.rear_left_motor, self.rear_right_motor]:
            motor.setPosition(float('inf'))
            motor.setVelocity(1)

        # ── STATE VARIABLES ────────────────────────────────────────────────
        self.current_pose    = 6 * [0]  # [X, Y, Z, roll, pitch, yaw]
        self.target_position = [0, 0, 0]  # Current waypoint [X, Y, bearing]
        self.target_index    = 0          # Index into patrol waypoint list

        # Camera-based fire tracking
        self.img_coord_fire  = []    # (cx, cy) of detected fire in image, or []
        self.WaterDropStatus = False # True while cooling down after water drop
        self.fire_miss_count = 0     # Consecutive detection failures

        print(f"[{self.getName()}] Initialized")

    # ── PERCEPTION ─────────────────────────────────────────────────────────

    def get_image_from_camera(self):
        """
        Capture and decode a frame from the downward camera.

        Uses getImage() (raw bytes) rather than getImageArray() (nested list)
        to avoid creating non-contiguous numpy arrays that crash OpenCV.

        Returns:
            numpy.ndarray: RGB image of shape (H, W, 3), or None if not ready
        """
        raw = self.camera.getImage()
        if not raw:
            return None   # Camera not yet initialised
        w   = self.camera.getWidth()
        h   = self.camera.getHeight()
        # Decode raw BGRA bytes → reshape to (H, W, 4) → convert to RGB
        img = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    def fire_detection(self):
        """
        Detect smoke/fire in the camera image using HSV colour segmentation.

        Algorithm:
            1. Convert RGB image to HSV colour space
            2. Apply threshold mask for smoke colour:
               H: 0–172 (any hue), S: 0–111 (low saturation = grey/white),
               V: 168–255 (bright) → captures the grey smoke above the fire
            3. Compute image moments to find centroid of all matching pixels
            4. Return centroid (cx, cy) if enough pixels detected

        Sensor fusion note:
            This provides image-space coordinates of the fire.
            Combined with GPS distance (world-space), the two sensors
            give complementary information: camera for visual confirmation,
            GPS for precise world-coordinate navigation.

        Returns:
            tuple (cx, cy): fire centroid in image pixels, or None if not found
        """
        try:
            img = self.get_image_from_camera()
            if img is None:
                return None
        except Exception as e:
            print(f"[{self.getName()}] Camera error: {e}")
            traceback.print_exc()
            return None

        # Convert to HSV for colour-based segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Smoke detection range: bright, low-saturation pixels (grey/white smoke)
        smoke_lower = np.array([0,   0,   168])
        smoke_upper = np.array([172, 111, 255])
        mask = cv2.inRange(hsv, smoke_lower, smoke_upper)

        # Ratio of smoke pixels to total image pixels
        fire_ratio = np.round(cv2.countNonZero(mask) / (img.size / 3) * 100, 2)

        if fire_ratio > 0.05:
            # Use image moments to find centroid of detected smoke region
            m = cv2.moments(mask)
            if m['m00'] > 50:   # Require at least 50 pixels (avoids noise)
                cx = m['m10'] / m['m00']
                cy = m['m01'] / m['m00']
                return (cx, cy)

        return None  # No significant smoke detected

    # ── NAVIGATION ─────────────────────────────────────────────────────────

    def set_position(self, pos):
        """Update the robot's current pose estimate from sensor readings."""
        self.current_pose = pos

    def move_to_target(self, waypoints, verbose_movement=False, verbose_target=False):
        """
        Navigate along a list of GPS waypoints in a looping patrol.

        Algorithm:
            1. Compute bearing from current GPS position to target waypoint
               using arctan2(dy, dx)
            2. Calculate angle_left = (bearing − current_yaw), normalised to (−π, π]
            3. Yaw disturbance ∝ angle_left → turns drone toward target
            4. Pitch disturbance = MAX_PITCH (constant forward lean)
            5. When within target_precision metres, advance to next waypoint

        Args:
            waypoints (list): List of [X, Y] world coordinates to patrol
            verbose_movement (bool): Print remaining angle and distance each step
            verbose_target (bool): Print when a waypoint is reached

        Returns:
            tuple: (yaw_disturbance, pitch_disturbance) motor control values
        """
        # Initialise target to first waypoint on first call
        if self.target_position[0:2] == [0, 0]:
            self.target_position[0:2] = waypoints[0]

        # Check if drone has reached current waypoint
        if all(abs(x1 - x2) < self.target_precision
               for x1, x2 in zip(self.target_position, self.current_pose[0:2])):
            self.target_index = (self.target_index + 1) % len(waypoints)
            self.target_position[0:2] = waypoints[self.target_index]

        # Compute bearing to target using GPS coordinates
        self.target_position[2] = np.arctan2(
            self.target_position[1] - self.current_pose[1],
            self.target_position[0] - self.current_pose[0])

        # Angle remaining to rotate (normalised to −π … +π)
        angle_left = self.target_position[2] - self.current_pose[5]
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        if angle_left > np.pi:
            angle_left -= 2 * np.pi

        # Proportional yaw control: larger angle → stronger turn command
        yaw_disturbance   = self.MAX_YAW_DISTURBANCE * angle_left / (2 * np.pi)
        pitch_disturbance = self.MAX_PITCH_DISTURBANCE  # Constant forward lean

        if verbose_movement:
            dist = np.sqrt((self.target_position[0] - self.current_pose[0])**2
                         + (self.target_position[1] - self.current_pose[1])**2)
            print(f"angle_left={angle_left:.3f} dist={dist:.2f}")

        return yaw_disturbance, pitch_disturbance

    # ── MAIN LOOP ──────────────────────────────────────────────────────────

    def run(self):
        """
        Main autonomous control loop — runs every simulation timestep.

        State machine:
            CLIMBING → reaches target altitude
            PATROL   → GPS waypoint following when no fire assigned
            FIRE     → GPS navigation to fire coordinates from supervisor

        Motor control (quadrotor differential):
            Positive pitch_input → front motors faster → nose pitches backward
            Negative pitch_input → rear motors faster  → nose pitches forward → forward motion
            Positive yaw_input   → front-right & rear-left faster → turn left (CCW)
            Negative yaw_input   → front-left & rear-right faster → turn right (CW)
        """
        # Timing accumulators for timed update blocks
        t1 = self.getTime()   # Motion update timer (every 0.1 s)
        t2 = self.getTime()   # Fire detection timer (every 1.0 s)
        t3 = self.getTime()   # Water cooldown timer

        # Disturbance values persist between timesteps (last commanded value held)
        roll_disturbance  = 0
        pitch_disturbance = 0
        yaw_disturbance   = 0

        # Parse patrol coordinates and target altitude from world file controller args
        parser = optparse.OptionParser()
        parser.add_option("--patrol_coords",  default="11 11, 11 21, 21 21, 21 11")
        parser.add_option("--target_altitude", default=42, type=float)
        options, _ = parser.parse_args()

        # Build waypoint list from comma-separated "x y" pairs
        waypoints = [[float(p.split()[0]), float(p.split()[1])]
                     for p in options.patrol_coords.split(',')]
        target_altitude = options.target_altitude

        print(f"[{self.getName()}] Patrolling {len(waypoints)} waypoints at {target_altitude}m")

        while self.step(self.time_step) != -1:
          try:
            # ── READ SENSORS ───────────────────────────────────────────────
            roll, pitch, yaw          = self.imu.getRollPitchYaw()
            Xpos, Ypos, altitude      = self.gps.getValues()
            roll_accel, pitch_accel, _ = self.gyro.getValues()
            self.set_position([Xpos, Ypos, altitude, roll, pitch, yaw])

            # ── READ SUPERVISOR FIRE ASSIGNMENT ────────────────────────────
            # Supervisor writes "fire:X:Y" when a fire is assigned to this drone
            fire_gps = None
            custom = self.getCustomData()
            if custom.startswith("fire:"):
                try:
                    parts = custom.split(":")
                    fire_gps = [float(parts[1]), float(parts[2])]
                except Exception:
                    fire_gps = None

            # ── FLIGHT CONTROL (only at patrol altitude) ───────────────────
            if altitude > target_altitude - 1:
                # Ramp pitch gradually on entry to patrol altitude — prevents
                # sudden nose-down that destabilises the drone
                if not hasattr(self, 'patrol_entry_time'):
                    self.patrol_entry_time = self.getTime()
                ramp = min((self.getTime() - self.patrol_entry_time) / 3.0, 1.0)

                if self.getTime() - t1 > 0.1:
                    if fire_gps:
                        # FIRE MODE: fly directly to fire GPS coordinates
                        dx = fire_gps[0] - Xpos
                        dy = fire_gps[1] - Ypos
                        dist          = np.sqrt(dx*dx + dy*dy)
                        target_bearing = np.arctan2(dy, dx)

                        angle_left = target_bearing - yaw
                        angle_left = (angle_left + 2*np.pi) % (2*np.pi)
                        if angle_left > np.pi:
                            angle_left -= 2*np.pi

                        # Proportional yaw toward fire; pitch scales down when
                        # facing wrong direction to reduce overshoot
                        yaw_disturbance   = self.MAX_YAW_DISTURBANCE * angle_left / (2*np.pi)
                        pitch_disturbance = (self.MAX_PITCH_DISTURBANCE * ramp *
                                             (1 - min(abs(angle_left), np.pi) / np.pi))

                        if dist > 4:
                            print(f"[{self.getName()}] Moving to fire at "
                                  f"({fire_gps[0]:.0f},{fire_gps[1]:.0f}) — {dist:.1f}m")
                    else:
                        # PATROL MODE: follow GPS waypoints around the forest
                        yaw_disturbance, pitch_disturbance = self.move_to_target(waypoints)
                    t1 = self.getTime()

                # ── CAMERA FIRE DETECTION (every 1 second) ─────────────────
                # Runs in background for sensor fusion; primary navigation uses GPS
                if self.getTime() - t2 > 1.0:
                    self.fire_detection()   # Result logged; GPS handles navigation
                    t2 = self.getTime()

            else:
                # CLIMBING: print altitude progress every 2 seconds
                if self.getTime() - t1 > 2:
                    print(f"[{self.getName()}] Climbing: {altitude:.1f}m → {target_altitude}m")
                    t1 = self.getTime()

            # ── WATER DROP ─────────────────────────────────────────────────
            if self.water_to_drop > 0:
                self.setCustomData(str(self.water_to_drop))
                self.water_to_drop = 0

            # ── PID MOTOR CONTROL ──────────────────────────────────────────
            # Roll and pitch stabilisation: sensor readings feed back into PID
            roll_input  = self.K_ROLL_P  * clamp(roll,  -1, 1) + roll_accel  + roll_disturbance
            pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_accel + pitch_disturbance
            yaw_input   = yaw_disturbance

            # Altitude control: cubic PID response for smooth approach to target
            alt_error      = clamp(target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
            vertical_input = self.K_VERTICAL_P * pow(alt_error, 3.0)

            # Differential motor speeds produce roll, pitch, yaw, and altitude
            # Note: front-right and rear-left spin opposite direction (negative) for torque balance
            self.front_left_motor.setVelocity(  self.K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input)
            self.front_right_motor.setVelocity(-(self.K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input))
            self.rear_left_motor.setVelocity(  -(self.K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input))
            self.rear_right_motor.setVelocity(   self.K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input)

          except Exception as e:
            print(f"[{self.getName()}] Runtime error: {e}")
            traceback.print_exc()


robot = Mavic()
robot.run()
