# Repository: https://github.com/Behdad-kanaani/Reinforcement-Learning-Grid-Navigation/
# Author: Behdad Kanaani
# Project Overview:
# This project showcases a **Reinforcement Learning (RL)** agent trained using the **Q-learning** algorithm 
# to navigate a grid-based environment. The agent starts at one of several positions and must find its way 
# to a goal while avoiding obstacles. Through repeated training episodes, the agent learns the most efficient 
# path by balancing **exploration** (trying random actions) and **exploitation** (choosing the best-known action). 
# The agent's decisions are guided by rewards and penalties, where reaching the goal gives a high reward, 
# hitting obstacles results in penalties, and each step also carries a small negative penalty to encourage efficiency.
# 
# **Key Highlights:**
# - The agent learns from experience by updating its Q-table, which stores the expected future rewards for each state-action pair.
# - The environment is visualized using **Pygame**, where obstacles are shown in red, the agent’s path in yellow, 
#   and the agent itself is represented by a blue square.
# 
# This project is a simple yet powerful example of **Q-learning** applied to pathfinding in a 2D grid. 
# It's a great way to explore reinforcement learning concepts in action.

import numpy as np
import random
import pygame

# Grid dimensions and tile size
GRID_WIDTH, GRID_HEIGHT = 15, 10
TILE_SIZE = 60

# Start positions and goal position
starts = [(0, 0), (14, 0), (0, 9), (14, 9)]
end = (8, 6)

# Obstacles in the grid
obstacles = [
    (1, 3), (4, 1), (6, 6), (8, 3), (3, 5), (7, 4),
    (10, 2), (12, 6), (11, 1), (13, 7), (5, 8),
    (7, 6), (2, 9), (6, 2), (1, 6), (2, 4), (4, 4),
    (5, 5), (6, 7), (8, 5), (9, 7), (10, 4), (11, 6)
]

# Available actions (down, up, right, left)
actions = [(0,1),(0,-1),(1,0),(-1,0)]
action_names = ["down", "up", "right", "left"]

# Initialize Q-table with zeros
Q = np.zeros((GRID_WIDTH, GRID_HEIGHT, len(actions)))

# Learning parameters
alpha = 0.5  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration probability
episodes = 100  # Number of training episodes

# Check if a position is valid (within grid and not an obstacle)
def is_valid(x, y):
    return 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT and (x, y) not in obstacles

# Training loop over multiple episodes
for ep in range(episodes):
    for start in starts:
        state = start
        while state != end:
            x, y = state

            # Exploration vs Exploitation: select action
            if random.uniform(0, 1) < epsilon:
                # Exploration: choose random action
                action_idx = random.randint(0, len(actions)-1)
            else:
                # Exploitation: choose best known action
                action_idx = np.argmax(Q[x, y])

            # Take the action and get the next state
            dx, dy = actions[action_idx]
            next_state = (x + dx, y + dy)

            # Calculate reward
            if next_state == end:
                reward = 100  # Reward for reaching the goal
            elif not is_valid(*next_state):
                reward = -10  # Penalty for hitting an obstacle
                next_state = state  # Stay in the same state
            else:
                reward = -1  # Penalty for each step

            # Update Q-value using the Q-learning formula
            nx, ny = next_state
            Q[x, y, action_idx] = Q[x, y, action_idx] + alpha * (reward + gamma * np.max(Q[nx, ny]) - Q[x, y, action_idx])
            
            # Move to the next state
            state = next_state

# Pathfinding: extract the learned path after training
state = start
path = [state]
while state != end:
    x, y = state
    action_idx = np.argmax(Q[x, y])  # Choose the best action based on the learned Q-table
    dx, dy = actions[action_idx]
    next_state = (x + dx, y + dy)
    if next_state == state:
        break  # Break if no movement
    path.append(next_state)
    state = next_state

# Print the learned path
print("Learned path by RL:")
print(path)

# Pygame setup for visualization
WIDTH, HEIGHT = GRID_WIDTH * TILE_SIZE, GRID_HEIGHT * TILE_SIZE
WHITE = (245, 245, 245)
GRAY = (200, 200, 200)
RED = (200, 50, 50)
BLUE = (50, 50, 200)
YELLOW = (255, 255, 0)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Path Visualization")
clock = pygame.time.Clock()

# Initialize player position at the start of the learned path
player_pos = list(path[0])
path_index = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move player through the path
    if path_index < len(path):
        player_pos = list(path[path_index])
        path_index += 1

    # Fill background with white
    screen.fill(WHITE)

    # Draw grid
    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            rect = pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, GRAY, rect, 1)

    # Draw obstacles
    for ox, oy in obstacles:
        rect = pygame.Rect(ox * TILE_SIZE, oy * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(screen, RED, rect)

    # Draw learned path
    for x, y in path:
        rect = pygame.Rect(x * TILE_SIZE + TILE_SIZE // 4, y * TILE_SIZE + TILE_SIZE // 4, TILE_SIZE // 2, TILE_SIZE // 2)
        pygame.draw.rect(screen, YELLOW, rect)

    # Draw the player (agent)
    px, py = player_pos
    rect = pygame.Rect(px * TILE_SIZE, py * TILE_SIZE, TILE_SIZE, TILE_SIZE)
    pygame.draw.rect(screen, BLUE, rect)

    # Update the screen
    pygame.display.flip()
    clock.tick(5)  # Set the speed of the agent's movement

pygame.quit()
