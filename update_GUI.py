# Writed By https://github.com/Behdad-kanaani

import numpy as np
import random
import pygame
import math
import sys

# ------------------------------
# Environment and Agent Classes
# ------------------------------
class GridWorld:
    """
    The GridWorld class is responsible for managing the environment and its rules.
    """
    def __init__(self, width, height, start, end, obstacles, actions):
        self.width = width
        self.height = height
        self.start = start
        self.end = end
        self.obstacles = set(obstacles)
        self.actions = actions

    def is_valid_state(self, x, y):
        """Checks if a position is valid within the grid."""
        return 0 <= x < self.width and 0 <= y < self.height and (x, y) not in self.obstacles

    def get_reward(self, state, next_state):
        """Returns the reward value based on the state transition."""
        if next_state == self.end:
            return 100
        if not self.is_valid_state(*next_state):
            return -10
        return -1

class QLearningAgent:
    """
    The QLearningAgent class is responsible for the Q-learning logic.
    """
    def __init__(self, env, alpha, gamma, epsilon, start_state):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.start_state = start_state
        self.current_state = start_state
        self.q_table = np.zeros((env.width, env.height, len(env.actions)))
        self.visited_trail = []

    def choose_action(self):
        """Selects an action based on the epsilon-greedy policy."""
        x, y = self.current_state
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, len(self.env.actions) - 1)
        else:
            return np.argmax(self.q_table[x, y])

    def take_step(self):
        """Takes a step in the environment and updates the Q-table."""
        action_idx = self.choose_action()
        dx, dy = self.env.actions[action_idx]
        
        x, y = self.current_state
        next_state = (x + dx, y + dy)
        
        reward = self.env.get_reward(self.current_state, next_state)

        if not self.env.is_valid_state(*next_state):
            next_state = self.current_state

        nx, ny = next_state
        self.q_table[x, y, action_idx] += self.alpha * (reward + self.gamma * np.max(self.q_table[nx, ny]) - self.q_table[x, y, action_idx])
        
        if next_state != self.current_state:
            self.visited_trail.append(next_state)
        
        self.current_state = next_state
        return next_state == self.env.end

    def find_best_path(self):
        """Computes the best path after training is complete."""
        path = [self.start_state]
        state = self.start_state
        while state != self.env.end:
            x, y = state
            action_idx = np.argmax(self.q_table[x, y])
            dx, dy = self.env.actions[action_idx]
            next_state = (x + dx, y + dy)
            if next_state == state: # Prevent infinite loops
                break
            path.append(next_state)
            state = next_state
        return path

    def reset(self):
        """Resets the agent's state for a new training episode."""
        self.current_state = self.start_state
        self.visited_trail.clear()

# ------------------------------
# Main Pygame Application Class
# ------------------------------
class GridWorldApp:
    def __init__(self):
        # Environment settings
        self.TILE_SIZE = 60
        self.GRID_WIDTH, self.GRID_HEIGHT = 15, 10
        self.starts = [(0, 0), (14, 0), (0, 9), (14, 9)]
        self.end_point = (8, 6)
        self.obstacles = [
            (1, 3), (4, 1), (6, 6), (8, 3), (3, 5), (7, 4),
            (10, 2), (12, 6), (11, 1), (13, 7), (5, 8),
            (7, 6), (2, 9), (6, 2), (1, 6), (2, 4), (4, 4),
            (5, 5), (6, 7), (8, 5), (9, 7), (10, 4), (11, 6)
        ]
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        # Learning settings
        self.alpha = 0.5
        self.gamma = 0.9
        self.epsilon = 0.1
        self.episodes = 50
        self.current_episode = 0

        # Pygame settings
        self.BUTTON_BAR_HEIGHT = 100
        self.WIDTH, self.HEIGHT = self.GRID_WIDTH * self.TILE_SIZE, self.GRID_HEIGHT * self.TILE_SIZE + self.BUTTON_BAR_HEIGHT + 20
        self.GRID_OFFSET_Y = 10 

        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Q-Learning Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        # Colors
        self.BG_COLOR = (5, 5, 8)
        self.GRID_COLOR = (20, 25, 40)
        self.OBSTACLE_COLOR = (220, 50, 50)
        self.GOAL_COLOR = (50, 255, 100)
        self.PLAYER_COLOR = (50, 150, 255)
        self.VISITED_COLOR_TRAIL = (150, 100, 255)
        self.PATH_COLOR = (100, 255, 255)
        self.TEXT_COLOR = (240, 240, 240)
        # رنگ دکمه‌ها روشن‌تر شد تا بهتر دیده شوند
        self.BUTTON_COLOR = (60, 75, 100)
        self.BUTTON_HOVER_COLOR = (80, 100, 130)
        self.BUTTON_TEXT_COLOR = (255, 255, 255)

        # Application states
        self.is_training = True
        self.selecting_start = False
        self.selecting_goal = False
        self.selecting_obstacle = False
        self.show_final_path = False
        
        # Speeds
        self.speeds = [40, 500, 1000]
        self.current_speed_index = 1

        # Instantiate classes
        self.env = GridWorld(self.GRID_WIDTH, self.GRID_HEIGHT, self.starts[0], self.end_point, self.obstacles, self.actions)
        self.agent = QLearningAgent(self.env, self.alpha, self.gamma, self.epsilon, self.env.start)
        self.final_path = []

        self.buttons = self.create_buttons()

    def create_buttons(self):
        """Generates button rects dynamically for a clean layout in two rows."""
        button_width = 150
        button_height = 40
        spacing = 15
        
        # Calculate positions for two rows
        row1_y = self.GRID_HEIGHT * self.TILE_SIZE + self.GRID_OFFSET_Y + spacing
        row2_y = row1_y + button_height + spacing

        # First row buttons
        total_row1_width = (3 * button_width) + (2 * spacing)
        start_x_row1 = (self.WIDTH - total_row1_width) / 2
        buttons = [
            {"text": "Select Start", "rect": pygame.Rect(start_x_row1, row1_y, button_width, button_height), "action": self.toggle_start_selection},
            {"text": "Select Goal", "rect": pygame.Rect(start_x_row1 + button_width + spacing, row1_y, button_width, button_height), "action": self.toggle_goal_selection},
            {"text": "Select Obstacles", "rect": pygame.Rect(start_x_row1 + 2 * (button_width + spacing), row1_y, button_width, button_height), "action": self.toggle_obstacle_selection},
        ]

        # Second row buttons
        total_row2_width = (3 * button_width) + (2 * spacing)
        start_x_row2 = (self.WIDTH - total_row2_width) / 2
        buttons.extend([
            {"text": "Show Final Path", "rect": pygame.Rect(start_x_row2, row2_y, button_width, button_height), "action": self.toggle_path_display},
            {"text": "Normal Speed", "rect": pygame.Rect(start_x_row2 + button_width + spacing, row2_y, button_width, button_height), "action": self.toggle_speed},
            {"text": "Random Obstacles", "rect": pygame.Rect(start_x_row2 + 2 * (button_width + spacing), row2_y, button_width, button_height), "action": self.set_random_obstacles},
            {"text": "Reset", "rect": pygame.Rect(self.WIDTH - button_width - 10, 10, button_width, button_height), "action": self.reset_simulation}
        ])
        return buttons

    # --- Event handling methods ---
    def toggle_start_selection(self):
        """Toggles the start point selection mode."""
        self.selecting_start = not self.selecting_start
        self.selecting_goal = False
        self.selecting_obstacle = False
        self.is_training = False

    def toggle_path_display(self):
        """Shows/hides the final path."""
        if not self.is_training and not self.selecting_start and not self.selecting_goal and not self.selecting_obstacle:
            self.show_final_path = not self.show_final_path
            self.buttons[3]["text"] = "Show Learning" if self.show_final_path else "Show Final Path"

    def toggle_speed(self):
        """Changes the simulation speed."""
        self.current_speed_index = (self.current_speed_index + 1) % len(self.speeds)
        speed_labels = ["Slow Speed", "Normal Speed", "Fast Speed"]
        self.buttons[4]["text"] = speed_labels[self.current_speed_index]

    def toggle_goal_selection(self):
        """Toggles the goal point selection mode."""
        self.selecting_goal = not self.selecting_goal
        self.selecting_start = False
        self.selecting_obstacle = False
        self.is_training = False
    
    def toggle_obstacle_selection(self):
        """Toggles the obstacle selection mode."""
        self.selecting_obstacle = not self.selecting_obstacle
        self.selecting_start = False
        self.selecting_goal = False
        self.is_training = False
        if not self.selecting_obstacle:
            # Reset simulation to learn a new path after changes
            self.reset_simulation(reset_obstacles=False)

    def set_random_obstacles(self):
        """Generates random obstacles and resets the simulation."""
        num_obstacles = random.randint(20, 30)
        new_obstacles = []
        for _ in range(num_obstacles):
            while True:
                obs_x = random.randint(0, self.GRID_WIDTH - 1)
                obs_y = random.randint(0, self.GRID_HEIGHT - 1)
                if (obs_x, obs_y) != self.env.start and (obs_x, obs_y) != self.env.end and (obs_x, obs_y) not in new_obstacles:
                    new_obstacles.append((obs_x, obs_y))
                    break
        self.env.obstacles = set(new_obstacles)
        self.reset_simulation(reset_obstacles=False)

    def reset_simulation(self, reset_obstacles=True):
        """Resets the entire simulation."""
        self.current_episode = 0
        self.is_training = True
        self.show_final_path = False
        self.final_path = []
        if reset_obstacles:
            self.env.obstacles = self.obstacles # Restore initial obstacles
        self.agent = QLearningAgent(self.env, self.alpha, self.gamma, self.epsilon, self.env.start)

    def handle_mouse_click(self, event):
        """Handles mouse clicks."""
        # Check for button clicks
        mouse_pos = event.pos
        for button in self.buttons:
            if button["rect"].collidepoint(mouse_pos):
                button["action"]()
                return

        # Handle point selection on the grid
        grid_y_start = self.GRID_OFFSET_Y
        grid_x = event.pos[0] // self.TILE_SIZE
        grid_y = (event.pos[1] - grid_y_start) // self.TILE_SIZE
        
        if self.selecting_start and self.env.is_valid_state(grid_x, grid_y) and (grid_x, grid_y) != self.env.end:
            self.env.start = (grid_x, grid_y)
            self.selecting_start = False
            self.reset_simulation(reset_obstacles=False)
        elif self.selecting_goal and self.env.is_valid_state(grid_x, grid_y) and (grid_x, grid_y) != self.env.start:
            self.env.end = (grid_x, grid_y)
            self.selecting_goal = False
            self.reset_simulation(reset_obstacles=False)
        elif self.selecting_obstacle and (grid_x, grid_y) != self.env.start and (grid_x, grid_y) != self.env.end:
            pos = (grid_x, grid_y)
            if pos in self.env.obstacles:
                self.env.obstacles.remove(pos)
            else:
                self.env.obstacles.add(pos)

    # --- Drawing methods ---
    def draw_text(self, text, position, color, centered=False):
        text_surface = self.font.render(text, True, color)
        if centered:
            text_rect = text_surface.get_rect(center=position)
            self.screen.blit(text_surface, text_rect)
        else:
            self.screen.blit(text_surface, position)

    def draw_button(self, rect, text, mouse_pos):
        color = self.BUTTON_HOVER_COLOR if rect.collidepoint(mouse_pos) else self.BUTTON_COLOR
        pygame.draw.rect(self.screen, color, rect, 0, 5)
        self.draw_text(text, rect.center, self.BUTTON_TEXT_COLOR, centered=True)

    def draw_3d_tile(self, x, y, color):
        rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE + self.GRID_OFFSET_Y, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, color, rect)
        # Add 3D effect (Highlights and shadows)
        highlight_color = (min(255, color[0] + 30), min(255, color[1] + 30), min(255, color[2] + 30))
        shadow_color = (max(0, color[0] - 10), max(0, color[1] - 10), max(0, color[2] - 10))
        pygame.draw.polygon(self.screen, highlight_color, [rect.topleft, rect.topright, (rect.topright[0]-5, rect.topright[1]+5), (rect.topleft[0]+5, rect.topleft[1]+5)])
        pygame.draw.polygon(self.screen, highlight_color, [rect.topleft, (rect.topleft[0]+5, rect.topleft[1]+5), (rect.bottomleft[0]+5, rect.bottomleft[1]-5), rect.bottomleft])
        pygame.draw.polygon(self.screen, shadow_color, [(rect.topright[0]-5, rect.topright[1]+5), rect.topright, rect.bottomright, (rect.bottomright[0]-5, rect.bottomright[1]-5)])
        pygame.draw.polygon(self.screen, shadow_color, [(rect.bottomleft[0]+5, rect.bottomleft[1]-5), rect.bottomleft, rect.bottomright, (rect.bottomright[0]-5, rect.bottomright[1]-5)])

    def draw_grid(self):
        for row in range(self.GRID_HEIGHT):
            for col in range(self.GRID_WIDTH):
                self.draw_3d_tile(col, row, self.GRID_COLOR)
            
    def draw_obstacles(self):
        margin = self.TILE_SIZE * 0.1
        for ox, oy in self.env.obstacles:
            obstacle_rect = pygame.Rect(ox * self.TILE_SIZE + margin, oy * self.TILE_SIZE + self.GRID_OFFSET_Y + margin, self.TILE_SIZE - 2*margin, self.TILE_SIZE - 2*margin)
            pygame.draw.rect(self.screen, self.OBSTACLE_COLOR, obstacle_rect, 0, 5)

    def draw_goal(self, pulse):
        gx, gy = self.env.end
        goal_surface = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        # Glow effect
        glow_radius = self.TILE_SIZE // 2 + int(5 * math.sin(pulse * 0.15))
        pygame.draw.circle(goal_surface, (50, 255, 100, 40), (self.TILE_SIZE // 2, self.TILE_SIZE // 2), glow_radius)
        pygame.draw.circle(goal_surface, self.GOAL_COLOR, (self.TILE_SIZE // 2, self.TILE_SIZE // 2), self.TILE_SIZE // 4)
        self.screen.blit(goal_surface, (gx * self.TILE_SIZE, gy * self.TILE_SIZE + self.GRID_OFFSET_Y))

    def draw_player(self, player_pos, pulse):
        px, py = player_pos
        glow_radius = self.TILE_SIZE // 2 + int(3 * math.sin(pulse))
        glow_surface = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (50, 150, 255, 100), (self.TILE_SIZE // 2, self.TILE_SIZE // 2), glow_radius)
        self.screen.blit(glow_surface, (px * self.TILE_SIZE, py * self.TILE_SIZE + self.GRID_OFFSET_Y))
        pygame.draw.circle(self.screen, self.PLAYER_COLOR,
                           (int(px * self.TILE_SIZE + self.TILE_SIZE / 2), int(py * self.TILE_SIZE + self.TILE_SIZE / 2) + self.GRID_OFFSET_Y),
                           self.TILE_SIZE // 3)

    def draw_fading_trail(self, visited_trail):
        if not visited_trail:
            return
        min_alpha = 255 * 0.3
        surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        for i, (vx, vy) in enumerate(visited_trail):
            alpha_val = int(min_alpha + (255 - min_alpha) * (i / (len(visited_trail) - 1 + 1e-6)))
            color = (self.VISITED_COLOR_TRAIL[0], self.VISITED_COLOR_TRAIL[1], self.VISITED_COLOR_TRAIL[2], alpha_val)
            pygame.draw.circle(surface, color,
                               (vx * self.TILE_SIZE + self.TILE_SIZE // 2, vy * self.TILE_SIZE + self.TILE_SIZE // 2 + self.GRID_OFFSET_Y),
                               self.TILE_SIZE // 6)
        self.screen.blit(surface, (0, 0))

    def draw_path_lines(self, path):
        if len(path) < 2: return
        for i in range(len(path) - 1):
            px, py = path[i]
            next_px, next_py = path[i+1]
            pygame.draw.line(self.screen, self.PATH_COLOR, 
                             (px * self.TILE_SIZE + self.TILE_SIZE // 2, py * self.TILE_SIZE + self.TILE_SIZE // 2 + self.GRID_OFFSET_Y), 
                             (next_px * self.TILE_SIZE + self.TILE_SIZE // 2, next_py * self.TILE_SIZE + self.TILE_SIZE // 2 + self.GRID_OFFSET_Y), 5)
        for px, py in path:
            pygame.draw.circle(self.screen, self.PATH_COLOR,
                               (px * self.TILE_SIZE + self.TILE_SIZE // 2, py * self.TILE_SIZE + self.TILE_SIZE // 2 + self.GRID_OFFSET_Y),
                               self.TILE_SIZE // 5)
    
    def draw_scene(self, pulse):
        """Draws all UI elements."""
        self.screen.fill(self.BG_COLOR)
        
        status_text = ""
        if self.selecting_start:
            status_text = "Click a tile to select a new start point."
        elif self.selecting_goal:
            status_text = "Click a tile to select a new goal."
        elif self.selecting_obstacle:
            status_text = "Click a tile to add or remove an obstacle."
        elif self.is_training:
            status_text = f"Episode: {self.current_episode}/{self.episodes}"
        else:
            status_text = "Final Path"
        
        # Draw status text at the top
        self.draw_text(status_text, (self.WIDTH // 2, 40), self.TEXT_COLOR, centered=True)
        
        self.draw_grid()
        self.draw_obstacles()
        self.draw_goal(pulse)
        
        if self.show_final_path and not self.is_training and not self.selecting_obstacle:
            self.draw_path_lines(self.final_path)
        elif self.is_training:
            self.draw_fading_trail(self.agent.visited_trail)
        
        self.draw_player(self.agent.current_state, pulse)

        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            self.draw_button(button["rect"], button["text"], mouse_pos)

    # --- Main application loop ---
    def run(self):
        """Runs the main application loop."""
        pulse = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_click(event)

            if self.is_training and not self.selecting_start and not self.selecting_goal and not self.selecting_obstacle:
                episode_finished = self.agent.take_step()
                if episode_finished:
                    self.current_episode += 1
                    if self.current_episode < self.episodes:
                        self.agent.reset()
                    else:
                        self.is_training = False
                        self.final_path = self.agent.find_best_path()

            pulse += 0.1
            self.draw_scene(pulse)
            
            pygame.display.flip()
            self.clock.tick(self.speeds[self.current_speed_index])

if __name__ == '__main__':
    app = GridWorldApp()
    app.run()
