# Forest Firefighting Robotics Simulation

**Course:** Computer Science  
**Platform:** Webots Simulator  
**Project:** Forest Firefighters

## Overview

This project implements an autonomous multi-robot system for detecting and extinguishing forest fires in a simulated environment using Webots. Two Mavic 2 Pro drones and a Spot legged robot work together to patrol a Sassafras forest, detect fires using camera-based perception, navigate to fire locations using GPS, and extinguish them by dropping water.

## Robots

### Mavic 2 Pro Drones (x2)
- Patrol the forest at 37m altitude using GPS waypoints
- Detect smoke and fire using OpenCV colour segmentation on camera images
- Navigate directly to fire GPS coordinates provided by the supervisor
- Drop water when positioned above the fire
- Split to handle multiple fires simultaneously

### Spot Legged Robot
- Operates at ground level in the forest
- Uses a diagonal trot gait for locomotion
- Navigated by supervisor using potential-field path planning (avoids trees)
- Sprays water on ground-level fires

## System Architecture

```
Fire Supervisor (fire.py)
├── Manages fire simulation (animation, growth, spreading)
├── Assigns fires to robots (one drone per fire)
├── Navigates Spot via potential-field
└── Triggers extinguishing when robots are close enough

Mavic Controller (autonomous_mavic.py)
├── PID altitude and attitude control
├── GPS-based navigation (patrol + fire approach)
└── Camera + OpenCV smoke detection

Spot Controller (spot.py)
└── Diagonal trot gait (FL+RR / FR+RL diagonal pairs)
```

## Key Algorithms

| Component | Method |
|---|---|
| Fire detection | HSV colour segmentation (OpenCV) |
| Drone navigation | GPS bearing + proportional yaw/pitch control |
| Drone altitude | Cubic PID controller |
| Spot locomotion | Diagonal trot gait (sinusoidal phase control) |
| Spot navigation | Potential-field (attraction + tree repulsion) |
| Fire assignment | Greedy one-to-one matching (nearest fire per drone) |
| Fire spreading | Local propagation within 6m radius every 5 seconds |

## How to Run

1. Open Webots R2021b
2. Open `worlds/forest_firefighters.wbt`
3. Press Play — simulation starts automatically

## Project Structure

```
forest_firefighters/
├── controllers/
│   ├── fire/              # Supervisor controller
│   ├── autonomous_mavic/  # Drone controller
│   └── spot/              # Spot robot controller
├── protos/                # Custom PROTO files (Fire, Smoke)
└── worlds/                # Webots world file
```

## Author

Namulinde Jean Madrine  
Computer Science
