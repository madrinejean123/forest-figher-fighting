# ============================================================
# Fire Supervisor Controller — Forest Firefighters Project
# Course: Computer Science
# ============================================================
# Webots Supervisor that manages the entire fire simulation.
# Handles fire animation, robot coordination, navigation,
# extinguishing logic, and fire lifecycle management.
# ============================================================

import json
import math
import random

from controller import Supervisor


# ── HELPER ──────────────────────────────────────────────────────────────────

def rotate(matrix, vector):
    """Apply a 3x3 rotation matrix to a 3D vector."""
    result = []
    for i in range(3):
        r = sum(matrix[3 * i + j] * vector[j] for j in range(3))
        result.append(r)
    return result


# ── TREE ────────────────────────────────────────────────────────────────────

class Tree:
    """
    Represents a single Sassafras tree in the forest.

    Tracks fire state, visual scale, and provides distance utilities.
    Each tree has a random robustness value that affects how easily
    fire spreads to it from neighbouring burning trees.
    """

    ROBUSTNESS_VARIATION = 2  # Max random resistance to fire spread

    def __init__(self, node):
        self.node        = node
        self.fire        = None     # Webots Fire PROTO node, or None
        self.smoke       = None     # Webots Smoke PROTO node, or None
        self.fire_count  = 0        # Number of simulation steps this tree has been burning
        self.fire_scale  = 1.0      # Current visual scale of the fire

        # Read permanent tree properties from the world
        self.translation = node.getField('translation').getSFVec3f()
        self.scale       = node.getField('size').getSFFloat() / 10

        # Random resistance — makes spread probabilistic per tree
        self.robustness  = random.uniform(0, self.ROBUSTNESS_VARIATION)

    def stopFire(self):
        """
        Extinguish fire and smoke on this tree.

        Both the Fire and Smoke PROTO nodes are hidden (moved off-screen
        to Y = 10,000,000) rather than deleted, since deletion during
        simulation can cause Webots instability.
        Reset fire_count so the tree can be re-ignited in future cycles.
        """
        if self.fire:
            tf = self.fire.getField('translation')
            t  = list(tf.getSFVec3f())
            t[1] = 10000000    # Move off-screen to hide
            tf.setSFVec3f(t)
            self.fire = None
        if self.smoke:
            sf = self.smoke.getField('translation')
            t  = list(sf.getSFVec3f())
            t[1] = 10000000
            sf.setSFVec3f(t)
            self.smoke = None

        # Reset for potential re-ignition in next fire cycle
        self.fire_count = 0
        self.fire_scale = 1.0

    def distance(self, coordinates):
        """3D Euclidean distance from this tree to a set of coordinates."""
        dx = self.translation[0] - coordinates[0]
        dy = self.translation[1] - coordinates[1]
        dz = self.translation[2] - coordinates[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)


# ── ROBOT WRAPPER ────────────────────────────────────────────────────────────

class Robot:
    """
    Supervisor-side wrapper for any robot node in the scene.

    Provides access to robot position, type, name, and water-dropping
    behaviour. Both Mavic drones and Spot are represented by this class.
    """

    MAX_WATER_RADIUS = 0.3

    def __init__(self, node):
        self.node          = node
        self.name          = node.getField('name').getSFString()
        self.type          = node.getTypeName()
        self.droppingWater = False
        self.waterBalls    = []

    def dropWater(self, children, quantity):
        """
        Spawn a Water node at the robot's current position.

        Water radius scales with quantity (more water = larger ball).
        Spot gets forward velocity on its water (spray effect).
        """
        position = self.node.getField('translation').getSFVec3f()
        radius   = min(self.MAX_WATER_RADIUS, 0.01 * quantity)
        water    = (f'Water {{ translation {position[0]} {position[1]} {position[2]} '
                    f'radius {radius} name "water {len(self.waterBalls)} {self.name}" }}')
        children.importMFNodeFromString(-1, water)
        water_node = children.getMFNode(-1)
        self.waterBalls.append(water_node)

        # Spot throws water forward with initial velocity
        if self.type == "Spot":
            orientation = rotate(self.node.getOrientation(), [0, 0, -1])
            velocity    = [v * 10 for v in orientation] + [0, 0, 0]
            water_node.setVelocity(velocity)

    def cleanWater(self):
        """Remove water balls that have fallen below the terrain (Z < 0)."""
        for ball in list(self.waterBalls):
            if ball.getField('translation').getSFVec3f()[2] < 0:
                self.waterBalls.remove(ball)
                ball.remove()

    def altitude(self):
        """Return the robot's current altitude (Z coordinate)."""
        return self.node.getField('translation').getSFVec3f()[2]


# ── WIND ─────────────────────────────────────────────────────────────────────

class Wind:
    """
    Simulates wind that biases the direction of fire propagation.

    Wind has intensity [0,1] and angle [0, 2π]. Both drift randomly
    each simulation step, creating realistic environmental variation.
    """

    INTENSITY_EVOLVE = 0.01   # Max change in intensity per step
    ANGLE_EVOLVE     = 0.005  # Max change in angle per step
    RANDOM_EVOLUTION = True   # Set False to freeze wind

    def __init__(self):
        self.intensity = random.random()
        self.angle     = random.uniform(0, 2 * math.pi)

    def evolve(self):
        """Randomly drift wind intensity and direction each step."""
        if self.RANDOM_EVOLUTION:
            self.intensity = max(0, min(1,
                self.intensity + self.INTENSITY_EVOLVE * random.uniform(-1, 1)))
            self.angle = (self.angle +
                self.ANGLE_EVOLVE * random.uniform(-2 * math.pi, 2 * math.pi)) % (2 * math.pi)

    def update(self, message):
        """Update wind from an external message string (JSON or "stop"/"start")."""
        if message == "stop":
            self.RANDOM_EVOLUTION = False
        elif message == "start":
            self.RANDOM_EVOLUTION = True
        else:
            wind        = json.loads(message)
            self.angle     = wind["angle"]
            self.intensity = wind["intensity"]

    def correctedDistance(self, tree1, tree2, propagation_radius):
        """
        Compute wind-corrected distance between two trees.

        Wind displaces the propagation centre from tree1, making fire
        spread further downwind than upwind.
        """
        x_wind = self.intensity * math.cos(self.angle)
        y_wind = self.intensity * math.sin(self.angle)
        dx = tree1.translation[0] + propagation_radius * x_wind - tree2.translation[0]
        dy = tree1.translation[1] + propagation_radius * y_wind - tree2.translation[1]
        dz = tree1.translation[2] - tree2.translation[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)


# ── FIRE SUPERVISOR ───────────────────────────────────────────────────────────

class Fire(Supervisor):
    """
    Supervisor controller for the forest fire simulation.

    Manages fire ignition, animation, spreading, and extinguishing.
    Coordinates drone and Spot robot assignments to ensure each fire
    is handled by a different robot. Uses proximity-based extinguishing
    (Spot: within 3m for 1s, drones: within 5m for 2s).
    """

    FLAME_CYCLE     = 13    # Fire PROTO has 13 animation frames
    FLAME_PEAK      = 17    # Unused (fire no longer self-extinguishes)
    MAX_PROPAGATION = 10    # Max spread radius in metres
    MAX_EXTINCTION  = 4     # Max distance for water-ball extinguishing

    def __init__(self):
        super(Fire, self).__init__()

        self.time_step   = int(self.getBasicTimeStep())
        self.update_fire = False
        self.wind        = Wind()

        root          = self.getRoot()
        self.children = root.getField('children')  # World root children (for spawning nodes)
        self.trees    = []
        self.robots   = []

        # Scan all world nodes to collect trees and robots
        n = self.children.getCount()
        for i in range(n):
            child      = self.children.getMFNode(i)
            child_name = child.getField('name')

            # Collect all Sassafras trees from the "uneven forest" solid
            if child_name and child_name.getSFString() == 'uneven forest':
                forest_children = child.getField('children')
                if forest_children:
                    for j in range(forest_children.getCount()):
                        fc = forest_children.getMFNode(j)
                        if fc.getTypeName() == 'Sassafras':
                            self.trees.append(Tree(fc))

            # Collect all robots (drones + Spot) that can extinguish fires
            if child.getBaseTypeName() == 'Robot':
                if (child.getField('translation') is not None and
                        child.getField('customData') is not None):
                    self.robots.append(Robot(child))

        print(f"Loaded {len(self.trees)} trees and {len(self.robots)} robots")

        # Ignite first fire near Spot so it has immediate work on startup
        if self.trees:
            spot = next((r for r in self.robots if r.node.getTypeName() == 'Spot'), None)
            if spot:
                sp      = spot.node.getField('translation').getSFVec3f()
                nearby  = [t for t in self.trees if math.sqrt(
                    (t.translation[0]-sp[0])**2 + (t.translation[1]-sp[1])**2) < 15]
                start   = random.choice(nearby) if nearby else random.choice(self.trees)
                print(f"Initial fire near Spot at ({start.translation[0]:.1f},{start.translation[1]:.1f})")
                self.ignite(start)

    # ── FIRE MECHANICS ───────────────────────────────────────────────────────

    def ignite(self, tree):
        """
        Spawn a Fire PROTO node at the given tree's location.

        The fire is scaled relative to the tree size for visual realism.
        References to the scale and translation fields are cached on the
        tree object to avoid repeated getField() calls in the burn loop.
        """
        if tree.fire_count > 1:
            return   # Already burning or recently extinguished

        root = self.getRoot().getField('children')
        s    = max(2.0, tree.scale * 5)   # Fire size proportional to tree, minimum 2m
        node = (f'Fire {{'
                f' translation {tree.translation[0]} {tree.translation[1]} {tree.translation[2]}'
                f' scale {s} {s} {s}'
                f'}}')
        root.importMFNodeFromString(-1, node)
        tree.fire                   = root.getMFNode(-1)
        tree.fire_scale             = s
        tree.fire_scale_field       = tree.fire.getField('scale')
        tree.fire_translation_field = tree.fire.getField('translation')

    def burn(self, tree):
        """
        Advance the fire animation for one simulation step.

        Animation technique:
            The Fire PROTO contains 13 flame texture frames stacked
            vertically at intervals of 100,000 local units. Cycling
            which frame is visible (by shifting the node's translation)
            creates the animated flame effect.

        Growth:
            Every FLAME_CYCLE steps, fire_scale grows by 20% until
            it reaches 6× the tree's scale — then stays constant.
            Fire never self-extinguishes; only robot intervention stops it.

        Smoke:
            On the first burn step, a Smoke PROTO is spawned at the
            tree's position. It grows gradually to become visible from
            the drones' patrol altitude, aiding camera-based detection.
        """
        self.update_fire = True
        tree.fire_count += 1

        # Grow fire scale every FLAME_CYCLE steps, up to a maximum
        max_scale = tree.scale * 6
        if tree.fire_count % self.FLAME_CYCLE == 0 and tree.fire_scale < max_scale:
            tree.fire_scale = min(max_scale, tree.fire_scale * 1.2)
            tree.fire_scale_field.setSFVec3f([tree.fire_scale] * 3)

        # Advance animation frame by shifting node Y-position
        # Each frame is 100,000 × fire_scale world units apart in Y
        t = list(tree.translation)
        t[1] -= 100000 * tree.fire_scale * (tree.fire_count % 13)
        tree.fire_translation_field.setSFVec3f(t)

        # Spawn smoke on the first burn step
        if tree.fire_count == 1:
            smoke = (f'Smoke {{ translation {tree.translation[0]} {tree.translation[1]}'
                     f' {tree.translation[2]} scale 0.01 0.01 0.01 }}')
            self.children.importMFNodeFromString(-1, smoke)
            tree.smoke                  = self.children.getMFNode(-1)
            tree.smoke_translation_field = tree.smoke.getField('translation')
            tree.smoke_scale_field       = tree.smoke.getField('scale')
            tree.smoke_translation       = tree.smoke_translation_field.getSFVec3f()

        # Grow smoke gradually to aid drone camera detection (visible from 37m altitude)
        if 0 < tree.fire_count < 70:
            s = tree.fire_count / 100
            tree.smoke_scale_field.setSFVec3f([s, s, s])

    def propagate(self, tree):
        """
        Attempt to spread fire from a burning tree to nearby trees.

        Uses wind-corrected distance so fire spreads further downwind.
        Each candidate tree has a random robustness that makes spreading
        probabilistic — not every nearby tree will catch fire.
        """
        # Fire strength ramps from 0 to 1 over the first 50 burn steps
        fire_strength      = min(tree.fire_count / 50, 1.0)
        propagation_radius = self.MAX_PROPAGATION * fire_strength

        for candidate in self.trees:
            if candidate == tree or candidate.fire:
                continue
            dist = self.wind.correctedDistance(tree, candidate, propagation_radius)
            # Spread if wind-adjusted distance is within radius minus robustness
            if dist + candidate.robustness < propagation_radius * math.sqrt(tree.scale):
                if random.random() < 0.15:   # 15% chance per eligible tree
                    self.ignite(candidate)

    # ── MAIN LOOP ────────────────────────────────────────────────────────────

    def run(self):
        """
        Main simulation loop — runs every timestep.

        Each step:
          1. Advance fire animation on all burning trees
          2. Pre-assign fires to drones (greedy one-to-one matching)
          3. Check drone-drone proximity and apply separation offset
          4. Navigate Spot using potential-field toward its nearest fire
          5. Check robot proximity to fires; extinguish when close enough
          6. Manage fire lifecycle: Spot cycle, drone fires, spreading, reset
        """
        # Persistent state initialised lazily with hasattr checks below
        while True:
            if self.step(self.time_step) == -1:
                break

            self.update_fire = True

            # ── STEP 1: BURN ACTIVE FIRES ─────────────────────────────────
            for tree in self.trees:
                if tree.fire:
                    self.burn(tree)

            # ── STEP 2: COLLECT STATE ─────────────────────────────────────
            if not hasattr(self, 'proximity_timers'):
                self.proximity_timers = {}
            if not hasattr(self, 'last_fire_pos'):
                self.last_fire_pos = None

            active_fires  = [t for t in self.trees if t.fire]
            drone_robots  = [r for r in self.robots if r.node.getTypeName() == 'Mavic2Pro']

            # ── STEP 3: FIRE-TO-DRONE ASSIGNMENT (one-to-one greedy) ──────
            # Each fire is assigned to exactly one drone (the nearest).
            # Remaining drones with no assignment return to patrol.
            # This prevents two drones going to the same fire.
            drone_assignments = {}
            remaining = list(active_fires)
            for drone in drone_robots:
                if not remaining:
                    break
                rpos     = drone.node.getField('translation').getSFVec3f()
                assigned = min(remaining, key=lambda t: math.sqrt(
                    (rpos[0]-t.translation[0])**2 + (rpos[1]-t.translation[1])**2))
                drone_assignments[drone.name] = assigned
                remaining.remove(assigned)   # Fire now taken; next drone gets a different one

            # ── STEP 5: ROBOT NAVIGATION AND EXTINGUISHING ────────────────
            for robot in self.robots:
                robot_pos = robot.node.getField('translation').getSFVec3f()
                is_spot   = robot.node.getTypeName() == 'Spot'

                if active_fires:
                    # Determine which fire this robot targets
                    if is_spot:
                        # Spot always goes to its nearest fire
                        nearest = min(active_fires, key=lambda t: math.sqrt(
                            (robot_pos[0]-t.translation[0])**2 + (robot_pos[1]-t.translation[1])**2))
                    else:
                        # Drones use pre-computed assignment; unassigned drones patrol
                        if robot.name not in drone_assignments:
                            robot.node.getField('customData').setSFString("patrol")
                            self.proximity_timers.pop(robot.name, None)
                            continue
                        nearest = drone_assignments[robot.name]

                    # Write fire GPS to robot's customData (drones read this for navigation)
                    robot.node.getField('customData').setSFString(
                        f"fire:{nearest.translation[0]:.1f}:{nearest.translation[1]:.1f}")

                    dx   = nearest.translation[0] - robot_pos[0]
                    dy   = nearest.translation[1] - robot_pos[1]
                    dist = math.sqrt(dx*dx + dy*dy)

                    # SPOT NAVIGATION: potential-field path planning
                    # Attraction toward fire + repulsion from trees within 3.5m
                    # This allows Spot to navigate around tree trunks automatically.
                    if is_spot and dist > 2:
                        speed = 0.8 * self.time_step / 1000   # 0.8 m/s converted to per-step

                        nav_x = dx / dist   # Unit vector toward fire (attraction)
                        nav_y = dy / dist

                        for tree in self.trees:
                            tx = robot_pos[0] - tree.translation[0]
                            ty = robot_pos[1] - tree.translation[1]
                            td = math.sqrt(tx*tx + ty*ty)
                            if 0.1 < td < 3.5:
                                # Repulsion strength increases as Spot gets closer to tree
                                strength = (3.5 - td) / (3.5 * td)
                                nav_x += tx * strength
                                nav_y += ty * strength

                        # Normalise and apply movement
                        nav_norm = max(math.sqrt(nav_x*nav_x + nav_y*nav_y), 0.01)
                        robot.node.getField('translation').setSFVec3f([
                            robot_pos[0] + (nav_x/nav_norm) * speed,
                            robot_pos[1] + (nav_y/nav_norm) * speed,
                            robot_pos[2]
                        ])

                    # PROXIMITY CHECK: trigger water drop and extinguish countdown
                    threshold       = 3 if is_spot else 5   # metres to trigger
                    extinguish_wait = 1 if is_spot else 2   # seconds until fire out
                    key             = robot.name

                    if dist < threshold:
                        if key not in self.proximity_timers:
                            # Robot just arrived — start timer and drop water
                            self.proximity_timers[key] = self.getTime()
                            label = "Spot" if is_spot else "Drone"
                            print(f"{label} {robot.name} arrived at fire!")

                            if is_spot:
                                # Spot sprays water forward at ground level
                                ti = int(self.getTime() * 10)
                                wx = robot_pos[0] + dx/max(dist, 0.1)
                                wy = robot_pos[1] + dy/max(dist, 0.1)
                                wz = robot_pos[2] + 1.0
                                w  = (f'Water {{ translation {wx:.1f} {wy:.1f} {wz:.1f} '
                                      f'radius 0.5 name "water_spot_{ti}" }}')
                                self.children.importMFNodeFromString(-1, w)
                                print(f"Spot spraying water!")
                            else:
                                # Drone drops water from altitude directly above fire
                                # with downward velocity for fast visual impact
                                ti = int(self.getTime() * 10)
                                w  = (f'Water {{ translation {nearest.translation[0]:.1f} '
                                      f'{nearest.translation[1]:.1f} {robot_pos[2]:.1f} '
                                      f'radius 0.25 name "water_{robot.name}_{ti}" }}')
                                self.children.importMFNodeFromString(-1, w)
                                water_node = self.children.getMFNode(-1)
                                water_node.setVelocity([0, 0, -20, 0, 0, 0])  # 20 m/s downward
                                print(f"Drone {robot.name} dropping water!")

                        # Spot sprays continuously every 3 seconds while on fire
                        if is_spot:
                            if not hasattr(nearest, 'spot_water_time'):
                                nearest.spot_water_time = 0
                            if self.getTime() - nearest.spot_water_time > 3.0:
                                ti = int(self.getTime() * 10)
                                wx = robot_pos[0] + dx/max(dist, 0.1)
                                wy = robot_pos[1] + dy/max(dist, 0.1)
                                w  = (f'Water {{ translation {wx:.1f} {wy:.1f} {robot_pos[2]+1.0:.1f} '
                                      f'radius 0.5 name "water_spot_{ti}" }}')
                                self.children.importMFNodeFromString(-1, w)
                                print(f"Spot continuing spray...")
                                nearest.spot_water_time = self.getTime()

                        # Check if robot has been on fire long enough to extinguish
                        if key in self.proximity_timers:
                            elapsed = self.getTime() - self.proximity_timers[key]
                            if elapsed > extinguish_wait:
                                self.last_fire_pos = [nearest.translation[0], nearest.translation[1]]
                                nearest.stopFire()
                                self.proximity_timers.pop(key, None)

                                # Update and display extinguish score
                                if not hasattr(self, 'fire_counter'):
                                    self.fire_counter = {}
                                self.fire_counter[robot.name] = self.fire_counter.get(robot.name, 0) + 1
                                scores = " | ".join(f"{k}: {v}" for k, v in self.fire_counter.items())
                                print(f"✅ {robot.name} extinguished fire! Score: [{scores}]")

                                if is_spot:
                                    # Schedule next Spot fire in 5 seconds (Spot gets no rest)
                                    self.next_spot_fire_time = self.getTime() + 5
                                    self.spot_pos_for_fire   = [robot_pos[0], robot_pos[1]]
                    else:
                        # Drone left proximity zone — clear timer (Spot timer persists)
                        if not is_spot:
                            self.proximity_timers.pop(key, None)

                else:
                    # No active fires — send robots to patrol
                    robot.node.getField('customData').setSFString("patrol")
                    self.proximity_timers.pop(robot.name, None)

            # ── STEP 6: FIRE LIFECYCLE MANAGEMENT ────────────────────────

            # SPOT FIRE CYCLE: spawn new fire near Spot 5s after it extinguishes
            if hasattr(self, 'next_spot_fire_time') and self.getTime() >= self.next_spot_fire_time:
                sp     = self.spot_pos_for_fire
                nearby = [t for t in self.trees if t.fire_count == 0 and math.sqrt(
                    (t.translation[0]-sp[0])**2 + (t.translation[1]-sp[1])**2) < 12]
                if nearby:
                    self.ignite(random.choice(nearby))
                    print("🔥 New fire for Spot!")
                del self.next_spot_fire_time
                del self.spot_pos_for_fire

            # PHASE 2: release drone fires once drones reach patrol altitude (>33m)
            if not hasattr(self, 'drone_fires_released'):
                self.drone_fires_released = False
            if not self.drone_fires_released:
                mavic_up = any(r.node.getField('translation').getSFVec3f()[2] > 33
                               for r in self.robots if r.node.getTypeName() == 'Mavic2Pro')
                if mavic_up:
                    self.drone_fires_released = True
                    # Choose trees away from Spot's area for drone fires
                    away = [t for t in self.trees if t.fire_count == 0 and math.sqrt(
                        (t.translation[0]-6.5)**2 + (t.translation[1]-3.25)**2) > 10]
                    n = random.choice([1, 2])   # 1 or 2 fires for drones
                    for _ in range(n):
                        if away:
                            chosen = random.choice(away)
                            away.remove(chosen)
                            self.ignite(chosen)
                    print(f"🚁 Drones at altitude — {n} fire(s) released!")

            # LOCAL SPREAD: every 5s spread to a tree within 6m of existing fire (max 3 total)
            if self.drone_fires_released and active_fires and len(active_fires) < 3:
                if not hasattr(self, 'next_spread_time'):
                    self.next_spread_time = self.getTime() + 5
                elif self.getTime() >= self.next_spread_time:
                    # Find trees close to any burning tree (local pack spread)
                    candidates = [t for burning in active_fires for t in self.trees
                                  if not t.fire and t.fire_count == 0 and
                                  math.sqrt((t.translation[0]-burning.translation[0])**2 +
                                            (t.translation[1]-burning.translation[1])**2) < 6]
                    if candidates:
                        self.ignite(random.choice(candidates))
                        print("🔥 Fire spreading locally!")
                    self.next_spread_time = self.getTime() + 5

            # DRONE FIRE CYCLE: 8s after drone extinguishes, new fire in drone area
            if hasattr(self, 'next_drone_fire_time') and self.getTime() >= self.next_drone_fire_time:
                drone_fires = len(active_fires)
                away  = [t for t in self.trees if t.fire_count == 0 and math.sqrt(
                    (t.translation[0]-6.5)**2 + (t.translation[1]-3.25)**2) > 10]
                count = min(getattr(self, 'next_drone_fire_count', 1), max(0, 2 - drone_fires))
                for _ in range(count):
                    if away:
                        chosen = random.choice(away)
                        away.remove(chosen)
                        self.ignite(chosen)
                if count == 2:
                    print("🔥🔥 Two fires for drones — split!")
                elif count == 1:
                    print("🔥 New fire for drones!")
                del self.next_drone_fire_time
                if hasattr(self, 'next_drone_fire_count'):
                    del self.next_drone_fire_count

            # FULL RESET: if all fires are out, restart entire cycle after 12s
            if not active_fires:
                if not hasattr(self, 'next_fire_time'):
                    self.next_fire_time = self.getTime() + 12
                    print("✅ All fires out — new cycle in 12 seconds")
                elif self.getTime() >= self.next_fire_time:
                    self.drone_fires_released = False
                    if hasattr(self, 'next_spread_time'):
                        del self.next_spread_time
                    # Restart near Spot so ground robot has immediate work
                    spot = next((r for r in self.robots if r.node.getTypeName() == 'Spot'), None)
                    if spot:
                        sp     = spot.node.getField('translation').getSFVec3f()
                        nearby = [t for t in self.trees if t.fire_count == 0 and math.sqrt(
                            (t.translation[0]-sp[0])**2 + (t.translation[1]-sp[1])**2) < 15]
                        self.ignite(random.choice(nearby) if nearby else
                                    random.choice([t for t in self.trees if t.fire_count == 0]))
                    print("🔥 New cycle started — Spot goes first!")
                    del self.next_fire_time
            elif hasattr(self, 'next_fire_time'):
                del self.next_fire_time


if __name__ == "__main__":
    fire = Fire()
    fire.run()
