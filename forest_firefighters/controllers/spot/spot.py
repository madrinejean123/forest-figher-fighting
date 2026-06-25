# ============================================================
# Spot Legged Robot Controller — Forest Firefighters Project
# Course: Computer Science
# ============================================================
# Ground-level firefighting robot using a diagonal trot gait.
# Receives fire coordinates from the supervisor via customData
# and trots toward the fire. Extinguishing is handled by the
# fire supervisor based on proximity detection.
# ============================================================

from controller import Robot
import math


class Spot(Robot):
    """
    Autonomous ground firefighting robot using a quadruped trot gait.

    Locomotion approach:
        Trot gait — two diagonal pairs alternate in anti-phase:
            Pair A: Front-Left (FL) + Rear-Right (RR)
            Pair B: Front-Right (FR) + Rear-Left (RL)
        While Pair A swings (foot in air), Pair B is in stance
        (foot on ground pushing body forward), and vice versa.

    Navigation:
        The fire supervisor writes fire world-coordinates into
        this robot's customData field ("fire:X:Y"). This controller
        reads that field and triggers the trot gait. The supervisor
        physically moves the robot body using potential-field
        navigation (attraction to fire + repulsion from trees).

    Motor layout (12 motors, 3 per leg):
        Index  0: FL shoulder abduction    Index  1: FL shoulder rotation    Index  2: FL elbow
        Index  3: FR shoulder abduction    Index  4: FR shoulder rotation    Index  5: FR elbow
        Index  6: RL shoulder abduction    Index  7: RL shoulder rotation    Index  8: RL elbow
        Index  9: RR shoulder abduction    Index 10: RR shoulder rotation    Index 11: RR elbow
    """

    def __init__(self):
        Robot.__init__(self)
        self.time_step = int(self.getBasicTimeStep())

        # All 12 joint motors in order: FL, FR, RL, RR (each: abduction, rotation, elbow)
        motor_names = [
            "front left shoulder abduction motor",  "front left shoulder rotation motor",  "front left elbow motor",
            "front right shoulder abduction motor", "front right shoulder rotation motor", "front right elbow motor",
            "rear left shoulder abduction motor",   "rear left shoulder rotation motor",   "rear left elbow motor",
            "rear right shoulder abduction motor",  "rear right shoulder rotation motor",  "rear right elbow motor"
        ]
        self.motors = [self.getDevice(n) for n in motor_names]

        # Cap motor velocity to prevent abrupt jerky movements
        for motor in self.motors:
            motor.setVelocity(5.0)

        print("[Spot] Initialized — trot gait ready")

    def trot(self, t):
        """
        Execute one timestep of the diagonal trot gait.

        Algorithm:
            Two sinusoidal signals in anti-phase drive the two diagonal pairs:
                phi_A = sin(omega * t)          → drives FL + RR
                phi_B = sin(omega * t + pi)     → drives FR + RL (opposite phase)

            For each leg:
                - Shoulder rotation = step_amplitude * phi
                  Positive phi → shoulder swings FORWARD (swing phase)
                  Negative phi → shoulder pushes BACKWARD (stance phase, propels body)

                - Elbow = lift_amplitude * max(0, phi)
                  Only positive phi raises the elbow (lifts foot during swing)
                  Negative phi → elbow stays at 0 (leg straight, bearing weight)

        This creates the characteristic trot: two legs swing while the
        opposite two push, alternating to produce forward locomotion.

        Args:
            t (float): Current simulation time in seconds
        """
        omega = 6.0   # Gait frequency (rad/s) — higher = faster stepping
        step  = 0.35  # Shoulder rotation amplitude (rad) — controls step length
        lift  = 0.45  # Elbow bend during swing (rad) — controls foot clearance

        phi_A = math.sin(omega * t)              # Phase for FL + RR pair
        phi_B = math.sin(omega * t + math.pi)    # Phase for FR + RL pair (anti-phase)

        # FRONT LEFT — Part of diagonal pair A
        self.motors[0].setPosition(-0.1)                  # Abduction: spread leg outward
        self.motors[1].setPosition(step * phi_A)          # Rotation: swing fwd / push back
        self.motors[2].setPosition(lift * max(0, phi_A))  # Elbow: lift foot during swing only

        # FRONT RIGHT — Part of diagonal pair B
        self.motors[3].setPosition(0.1)                   # Abduction: spread leg outward
        self.motors[4].setPosition(step * phi_B)
        self.motors[5].setPosition(lift * max(0, phi_B))

        # REAR LEFT — Part of diagonal pair B (same phase as FR)
        self.motors[6].setPosition(-0.1)
        self.motors[7].setPosition(step * phi_B)
        self.motors[8].setPosition(lift * max(0, phi_B))

        # REAR RIGHT — Part of diagonal pair A (same phase as FL)
        self.motors[9].setPosition(0.1)
        self.motors[10].setPosition(step * phi_A)
        self.motors[11].setPosition(lift * max(0, phi_A))

    def stand(self):
        """
        Set all joints to a stable standing position.

        The shoulder abduction motors MUST be at ±0.1 rad to spread the
        legs outward. If all motors are set to 0, the legs collapse
        inward and the robot tips over.
        """
        # [abduction, rotation, elbow] for each leg in order FL, FR, RL, RR
        positions = [
            -0.1, 0, 0,   # Front Left
             0.1, 0, 0,   # Front Right
            -0.1, 0, 0,   # Rear Left
             0.1, 0, 0    # Rear Right
        ]
        for i, pos in enumerate(positions):
            self.motors[i].setPosition(pos)

    def run(self):
        """
        Main control loop.

        Each timestep:
          1. Read customData from supervisor (format: "fire:X:Y" or "patrol")
          2. If fire detected: execute trot gait (locomotion toward fire)
          3. If no fire: hold standing position

        The supervisor handles:
          - Detecting which fire is nearest to Spot
          - Moving Spot's body position using potential-field navigation
          - Triggering water spray and fire extinguishing
        """
        last_print = 0

        while self.step(self.time_step) != -1:
            t = self.getTime()
            custom = self.getCustomData()   # Read fire coordinates from supervisor

            if custom.startswith("fire:"):
                # Active fire assigned — execute trot gait for locomotion
                self.trot(t)
                if t - last_print > 2.0:
                    try:
                        parts = custom.split(":")
                        fx, fy = float(parts[1]), float(parts[2])
                        print(f"[Spot] Trotting to fire at ({fx:.0f}, {fy:.0f})")
                    except Exception:
                        pass
                    last_print = t
            else:
                # No active fire — stand still and wait
                self.stand()


robot = Spot()
robot.run()
