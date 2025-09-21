# 🧠 Reinforcement Learning: Grid Navigation Agent

![Project Demo](https://github.com/Behdad-kanaani/Reinforcement-Learning-Grid-Navigation/blob/main/demo.png)
*Watch the agent learn to navigate the grid in real-time!*

A hands-on **Reinforcement Learning (RL)** project demonstrating how an agent learns to navigate a 2D grid using the **Q-learning algorithm**. The goal is to reach the target efficiently while avoiding obstacles. Perfect for learning RL concepts visually and interactively!

---

## ✨ Key Features

* **🖱 Interactive Controls:** Click to set start/goal positions and add/remove obstacles.
* **📈 Real-time Learning:** Observe the agent exploring and improving its strategy dynamically.
* **🎲 Randomized Environments:** Test adaptability with random obstacle layouts.
* **⏩ Adjustable Speed:** Control simulation speed to analyze the learning process step by step.
* **🌙 Modern Dark Theme:** Clean interface with subtle animations for a professional feel.

---

## 🧩 How It Works

The project consists of **two main components**:

### 1️⃣ Q-learning Core

* Implements the Q-learning algorithm.
* Rewards the agent for reaching the goal.
* Penalizes collisions and unnecessary steps.
* Continuously updates the Q-table to improve performance.

### 2️⃣ Interactive GUI (Pygame)

* Visualizes the agent’s learning journey in real-time.
* Provides an interactive environment for experimentation.
* Dynamic updates make learning both educational and fun.

---

## ⚙️ Prerequisites

* Python 3.x
* [NumPy](https://numpy.org/)
* [Pygame](https://www.pygame.org/news)

---

## 🛠 Installation

1. Clone the repository:

```bash
git clone https://github.com/Behdad-kanaani/Reinforcement-Learning-Grid-Navigation.git
cd Reinforcement-Learning-Grid-Navigation
```

2. Install dependencies:

```bash
pip install numpy pygame
```

---

## ▶️ Running the Simulation

```bash
python update_GUI.py
```

* Set start and goal positions.
* Add/remove obstacles dynamically.
* Adjust simulation speed and watch the agent learn in real-time.

---

## 📚 Learning Outcomes

* Visual understanding of **Q-learning**.
* Experiment with **exploration vs. exploitation** strategies.
* Observe how **reward shaping** affects learning efficiency.
* Hands-on experience with **reinforcement learning algorithms**.

---

## 🏆 Highlights

* Fully interactive, beginner-friendly RL environment.
* Perfect for demos, teaching, and experimentation.
* Easily extendable for research or advanced RL projects.
* Modern, visually appealing interface for presentations.

---

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See the [LICENSE](LICENSE) file for details.

---

💡 **Pro Tip:** Experiment with different learning rates and discount factors in `update_GUI.py` to see how the agent’s behavior changes!
