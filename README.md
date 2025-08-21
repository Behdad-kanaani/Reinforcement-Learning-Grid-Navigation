

# 🧠 Reinforcement Learning Grid Navigation 🚀

## 🌟 Project Overview

This project demonstrates a **Reinforcement Learning (RL)** agent navigating a grid environment. The agent uses **Q-learning** to learn the optimal path from several possible start positions to a goal, while avoiding obstacles. The environment is visualized using **Pygame**, where obstacles are shown in red, the agent’s learned path in yellow, and the agent itself is represented by a blue square.

## 🔥 Features

* **Grid Environment**: The grid is customizable with configurable width, height, and tile size. Obstacles are randomly placed throughout the grid.
* **Q-Learning Agent**: The agent uses Q-learning to determine the best path to the goal while avoiding obstacles. The Q-table is updated based on rewards for valid moves, penalties for hitting obstacles, and rewards for reaching the goal.
* **Exploration vs Exploitation**: The agent follows an **epsilon-greedy** strategy, balancing between **exploration** (random actions to explore new paths) and **exploitation** (choosing the best-known path based on learned Q-values).
* **Multiple Start Positions**: The agent can start from several positions on the grid, offering diverse learning opportunities.
* **Obstacle Handling**: The agent learns to avoid obstacles and is penalized for invalid moves.
* **Visualization**: The agent’s learning process and pathfinding are visualized in real-time using Pygame.

## 📦 Requirements

* Python 3.x
* `pygame`
* `numpy`

Install the dependencies by running the following:

```bash
pip install -r requirements.txt
```

## 🚀 Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/Reinforcement-Learning-Grid-Navigation.git
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main script**:

   ```bash
   python main.py
   ```

Once you run the script, the agent will begin learning and navigating the grid. The entire process will be visualized, showing the agent’s movement, obstacles, and the learned path.

## 📖 Code Explanation

### Q-Table

The Q-table is a 3D NumPy array where each grid cell holds Q-values corresponding to each possible action (down, up, right, left). These values are updated during training, guiding the agent towards better decisions over time.

### Training Loop

The agent starts from one of the selected start positions and learns over multiple episodes. During each episode, the agent selects actions based on an epsilon-greedy policy, updates the Q-table, and learns from the rewards and penalties it receives.

### Visualization

After training, the agent’s learned path is displayed in real-time, showing its movements step by step. This allows you to track the agent’s progress and understand its decision-making process.

## 🖼️ Visual Representation

* **White Grid**: Empty grid cells where the agent can move.
* **Red Squares**: Obstacles that the agent must avoid.
* **Yellow Path**: The learned path of the agent.
* **Blue Square**: The agent’s current position.
se. Feel free to modify and use it for educational purposes!

## 🙏 Acknowledgments

* Special thanks to the **Pygame** library for providing the necessary tools to visualize the grid and agent's movements.
* This project was built for educational purposes to demonstrate reinforcement learning in a grid environment.

### This is a personal training project for me, and I hope it can be helpful to you as well. 😊 Thank you for your support! 🙏
