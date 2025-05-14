# ARS NEAT Maze Navigation Simulator

A Python project that evolves neural‐network controllers (via NEAT) for a differential‐drive robot to navigate randomly generated mazes using only range sensors and a Kalman filter.

---

## Description

This simulator evolves feed-forward neural networks (using the NEAT algorithm) to control a two-wheeled robot through randomly generated mazes.  

It combines the four parts of the assigment:

- **Simulation**
- **Kalman filter** for state estimation  
- **Occupancy grid mapping** for mapping
- **Parallel NEAT training** for robotic behavoiur  

---


## Installation
Install dependencies:
pip install -r requirements.txt

## Quick Start

### Manual mode (keyboard control):
python Simulation.py

Select mode [1] Manual when prompted

### Train with NEAT:
python Simulation.py

Select mode [2] NEAT Training


## Usage
### Manual Mode
- W(forward)/S(backward) for left wheel, O(forward)/L(backward) for right wheel control
- R: reset maze
- Esc: quit

### Automated Mode
- Adjust NEAT parameters in neat-config.txt
- Training runs for 50 generations by default
- Checkpoints saved every 5 generations (files neat-checkpoint-*)
