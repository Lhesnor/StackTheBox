# Stack the box

![Preview](assets/preview_gif.gif)

A custom multi-agent environment built with **PettingZoo**, **Box2D**, and **PyGame**.  
Agents are boxes that can move left/right and jump to build towers.  
The environment supports multiple reward modes for different training setups (sparse, dense, stable height, etc.).

## Features
- Multi-agent `ParallelEnv` API (compatible with PettingZoo & RLlib).
- Several reward schemes (`highest`, `dense_height`, `dense_height_stable_sq`, etc.).
- Configurable physics (gravity, friction, damping, etc.).
- Rendering with animated emotion sprites for boxes.
- Supports parameter-sharing training in RLlib.

## Installation
```bash
pip install "ray[rllib]==2.34.0" pettingzoo gymnasium numpy pygame Box2D
```

## Training
Run PPO training with RLlib:
```bash
python main.py --workers 4 --timesteps 1000000
```