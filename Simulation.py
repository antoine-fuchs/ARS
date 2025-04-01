import pygame
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ball Collision Game")

# Define colors
green = (0, 255, 0)
red = (255, 0, 0)
black = (0, 0, 0)
yellow = (255, 255, 0)
white = (255, 255, 255)  # Color for direction arrow

# Ball properties
ball_radius = 20
ball_x, ball_y = WIDTH // 2, HEIGHT // 2
ball_speed = 4

# Obstacle properties (a circular obstacle)
obstacle_x, obstacle_y = WIDTH // 3, HEIGHT // 3
obstacle_radius = 40

# Ball movement
velocity_x, velocity_y = 0, 0

# Game loop control
running = True
clock = pygame.time.Clock()
collision = False

while running:
    screen.fill(black)  # Clear the screen
    
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                velocity_x = -1
            elif event.key == pygame.K_RIGHT:
                velocity_x = 1
            elif event.key == pygame.K_UP:
                velocity_y = -1
            elif event.key == pygame.K_DOWN:
                velocity_y = 1
        elif event.type == pygame.KEYUP:
            if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                velocity_x = 0
            if event.key in (pygame.K_UP, pygame.K_DOWN):
                velocity_y = 0
    
    # Normalize speed for uniform diagonal movement
    if velocity_x != 0 or velocity_y != 0:
        length = math.sqrt(velocity_x**2 + velocity_y**2)
        velocity_x = (velocity_x / length) * ball_speed
        velocity_y = (velocity_y / length) * ball_speed

    # Calculate new position
    new_x = ball_x + velocity_x
    new_y = ball_y + velocity_y

    # Check collision with the obstacle (circle collision detection)
    ball_center = pygame.math.Vector2(new_x, new_y)
    obstacle_center = pygame.math.Vector2(obstacle_x, obstacle_y)
    distance = ball_center.distance_to(obstacle_center)
    
    if distance < ball_radius + obstacle_radius:
        collision = True
        direction = (ball_center - obstacle_center).normalize()
        new_x = obstacle_center.x + direction.x * (ball_radius + obstacle_radius)
        new_y = obstacle_center.y + direction.y * (ball_radius + obstacle_radius)
    else:
        collision = False
        ball_x, ball_y = new_x, new_y

    # Check collision with walls
    ball_x = max(ball_radius, min(WIDTH - ball_radius, ball_x))
    ball_y = max(ball_radius, min(HEIGHT - ball_radius, ball_y))

    # Draw obstacle
    pygame.draw.circle(screen, red, (obstacle_x, obstacle_y), obstacle_radius)
    
    # Change ball color if collision occurs
    ball_color = yellow if collision else green
    pygame.draw.circle(screen, ball_color, (int(ball_x), int(ball_y)), ball_radius)
    
    # Draw direction indicator
    if velocity_x != 0 or velocity_y != 0:
        direction_vector = pygame.math.Vector2(velocity_x, velocity_y)
        if direction_vector.length() != 0:
            direction_vector = direction_vector.normalize() * 20  # Normalized length of 20 pixels
        end_x = ball_x + direction_vector.x
        end_y = ball_y + direction_vector.y
        pygame.draw.line(screen, white, (ball_x, ball_y), (end_x, end_y), 3)
    
    pygame.display.flip()
    clock.tick(30)  # Limit frame rate to 30 FPS

pygame.quit()
