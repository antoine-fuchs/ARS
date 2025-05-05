import pygame
# Colors
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255,255,255)
BLACK = (0,0,0)

# Screen dimensions and grid size
WIDTH, HEIGHT = 800, 800
COLS, ROWS = 10, 10
CELL_SIZE = WIDTH // COLS
# Ball properties
ball_radius = min(CELL_SIZE // 2 - 4, 15)  # Ball must be smaller than a cell
start_x, start_y = CELL_SIZE // 2, CELL_SIZE // 2  # Start position
ball_x, ball_y = start_x, start_y  # Current position
ball_angle = 0  # Facing direction in radians

# Target ball properties
target_x, target_y = 0, 0  # Will be set randomly

# Game states
TARGET_REACHED = False
RETURNING_TO_START = False
GAME_COMPLETED = False

# Wheel speeds
left_wheel_speed = 0
right_wheel_speed = 0
wheel_max_speed = 10
wheel_base = 20  # Distance between wheels

# Wall thickness for collision detection
WALL_THICKNESS = 2

  # a sensor at every 30 degrees
num_sensors = 12
sensor_angles = [i * 30 for i in range(num_sensors)]


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Differential Drive Ball Simulation")

font_maze_id = pygame.font.SysFont(None, 12)
font_sensors = pygame.font.SysFont(None, 12)
font_hints = pygame.font.SysFont(None, 24)
font_speed = pygame.font.SysFont(None, 12)