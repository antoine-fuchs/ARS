import pygame
import math
import random
from maze import generate_maze, Cell, CELL_SIZE, WIDTH, HEIGHT, BLACK, WHITE

# Initialize Pygame
pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Differential Drive Ball Simulation")

# Colors
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)

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
wheel_max_speed = 2
wheel_base = 20  # Distance between wheels

# Wall thickness for collision detection
WALL_THICKNESS = 2

# Circle-Rectangle collision detection
def circle_rect_collision(circle_x, circle_y, radius, rect_x, rect_y, rect_width, rect_height):
    # Calculate the closest point on the rectangle to the circle center
    closest_x = max(rect_x, min(circle_x, rect_x + rect_width))
    closest_y = max(rect_y, min(circle_y, rect_y + rect_height))
    
    # Calculate the distance between the closest point and the circle center
    distance_x = circle_x - closest_x
    distance_y = circle_y - closest_y
    
    # Calculate squared distance
    distance_squared = distance_x * distance_x + distance_y * distance_y
    
    # If distance is less than or equal to radius, there is a collision
    return distance_squared <= (radius * radius)

# Circle-Circle collision detection
def circle_circle_collision(x1, y1, r1, x2, y2, r2):
    distance_squared = (x1 - x2) ** 2 + (y1 - y2) ** 2
    return distance_squared <= ((r1 + r2) ** 2)

# Check for collision with maze walls
def check_wall_collision(ball_x, ball_y, ball_radius, grid):
    for cell in grid:
        x = cell.x * CELL_SIZE
        y = cell.y * CELL_SIZE
        
        # Check each wall of the cell
        if cell.walls[0]:  # Top wall
            rect = pygame.Rect(x, y, CELL_SIZE, WALL_THICKNESS)
            if circle_rect_collision(ball_x, ball_y, ball_radius, rect.x, rect.y, rect.width, rect.height):
                return True, 'top'
        
        if cell.walls[1]:  # Right wall
            rect = pygame.Rect(x + CELL_SIZE - WALL_THICKNESS, y, WALL_THICKNESS, CELL_SIZE)
            if circle_rect_collision(ball_x, ball_y, ball_radius, rect.x, rect.y, rect.width, rect.height):
                return True, 'right'
        
        if cell.walls[2]:  # Bottom wall
            rect = pygame.Rect(x, y + CELL_SIZE - WALL_THICKNESS, CELL_SIZE, WALL_THICKNESS)
            if circle_rect_collision(ball_x, ball_y, ball_radius, rect.x, rect.y, rect.width, rect.height):
                return True, 'bottom'
        
        if cell.walls[3]:  # Left wall
            rect = pygame.Rect(x, y, WALL_THICKNESS, CELL_SIZE)
            if circle_rect_collision(ball_x, ball_y, ball_radius, rect.x, rect.y, rect.width, rect.height):
                return True, 'left'
    
    return False, None

def adjust_ball_position(ball_x, ball_y, ball_radius, grid):
    """Adjust the ball position so it exactly touches walls when in collision"""
    collision, wall_type = check_wall_collision(ball_x, ball_y, ball_radius, grid)
    
    if not collision:
        return ball_x, ball_y
    
    # Find the cell that has collision
    for cell in grid:
        x = cell.x * CELL_SIZE
        y = cell.y * CELL_SIZE
        
        if cell.walls[0] and wall_type == 'top':  # Top wall
            rect = pygame.Rect(x, y, CELL_SIZE, WALL_THICKNESS)
            if circle_rect_collision(ball_x, ball_y, ball_radius, rect.x, rect.y, rect.width, rect.height):
                return ball_x, y + WALL_THICKNESS - ball_radius
                
        if cell.walls[1] and wall_type == 'right':  # Right wall
            rect = pygame.Rect(x + CELL_SIZE - WALL_THICKNESS, y, WALL_THICKNESS, CELL_SIZE)
            if circle_rect_collision(ball_x, ball_y, ball_radius, rect.x, rect.y, rect.width, rect.height):
                return x + CELL_SIZE + WALL_THICKNESS - ball_radius, ball_y
                
        if cell.walls[2] and wall_type == 'bottom':  # Bottom wall
            rect = pygame.Rect(x, y + CELL_SIZE - WALL_THICKNESS, CELL_SIZE, WALL_THICKNESS)
            if circle_rect_collision(ball_x, ball_y, ball_radius, rect.x, rect.y, rect.width, rect.height):
                return ball_x, y - CELL_SIZE - WALL_THICKNESS - ball_radius
                
        if cell.walls[3] and wall_type == 'left':  # Left wall
            rect = pygame.Rect(x, y, WALL_THICKNESS, CELL_SIZE)
            if circle_rect_collision(ball_x, ball_y, ball_radius, rect.x, rect.y, rect.width, rect.height):
                return x + WALL_THICKNESS - ball_radius, ball_y
    
    return ball_x, ball_y

# Function to place target randomly in a valid position
def place_target_randomly(grid, player_x, player_y, ball_radius):
    while True:
        # Choose a random cell
        random_cell = random.choice(grid)
        # Calculate center of the cell
        cell_center_x = (random_cell.x * CELL_SIZE) + (CELL_SIZE // 2)
        cell_center_y = (random_cell.y * CELL_SIZE) + (CELL_SIZE // 2)
        
        # Check if it's not too close to the player's starting position
        min_distance = 5 * CELL_SIZE  # Ensure target is at least 5 cells away
        if ((cell_center_x - player_x) ** 2 + (cell_center_y - player_y) ** 2) >= (min_distance ** 2):
            # Check if it doesn't collide with walls
            if not check_wall_collision(cell_center_x, cell_center_y, ball_radius, grid)[0]:
                return cell_center_x, cell_center_y

def main():
    # Generate the maze
    grid = generate_maze()

    clock = pygame.time.Clock()
    running = True

    global left_wheel_speed, right_wheel_speed, ball_x, ball_y, ball_angle
    global TARGET_REACHED, RETURNING_TO_START, GAME_COMPLETED

    # Place the target ball
    target_x, target_y = place_target_randomly(grid, start_x, start_y, ball_radius)

    trail = []

    while running:
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    left_wheel_speed = wheel_max_speed
                elif event.key == pygame.K_s:
                    left_wheel_speed = -wheel_max_speed
                elif event.key == pygame.K_o:
                    right_wheel_speed = wheel_max_speed
                elif event.key == pygame.K_l:
                    right_wheel_speed = -wheel_max_speed
                elif event.key == pygame.K_r:
                    # Reset game
                    ball_x, ball_y = start_x, start_y
                    ball_angle = 0
                    TARGET_REACHED = False
                    RETURNING_TO_START = False
                    GAME_COMPLETED = False
                    target_x, target_y = place_target_randomly(grid, start_x, start_y, ball_radius)
                    trail.clear()

            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_w, pygame.K_s):
                    left_wheel_speed = 0
                if event.key in (pygame.K_o, pygame.K_l):
                    right_wheel_speed = 0

        # Differential drive kinematics
        v = (right_wheel_speed + left_wheel_speed) / 2
        omega = (right_wheel_speed - left_wheel_speed) / wheel_base

        # Update angle
        ball_angle += omega

        # Desired deltas
        dx = v * math.cos(ball_angle)
        dy = v * math.sin(ball_angle)

        # Move in small steps to get as close as possible
        steps = int(max(abs(dx), abs(dy)) / 0.1) + 1
        step_dx = dx / steps
        step_dy = dy / steps

        for _ in range(steps):
            # Try moving in X
            if not check_wall_collision(ball_x + step_dx, ball_y, ball_radius, grid)[0]:
                ball_x += step_dx
            else:
                break  # Stop when collision occurs

        for _ in range(steps):
            # Try moving in Y
            if not check_wall_collision(ball_x, ball_y + step_dy, ball_radius, grid)[0]:
                ball_y += step_dy
            else:
                break  # Stop when collision occurs

        # Check collision with target
        if not TARGET_REACHED and circle_circle_collision(ball_x, ball_y, ball_radius, target_x, target_y, ball_radius):
            TARGET_REACHED = True
            RETURNING_TO_START = True
        
        # Check if returned to start after reaching target
        if RETURNING_TO_START and circle_circle_collision(ball_x, ball_y, ball_radius, start_x, start_y, ball_radius):
            GAME_COMPLETED = True
            RETURNING_TO_START = False

        # Final collision state (for coloring)
        collision, _ = check_wall_collision(ball_x, ball_y, ball_radius, grid)

        # Clamp inside screen
        ball_x = max(ball_radius, min(WIDTH - ball_radius, ball_x))
        ball_y = max(ball_radius, min(HEIGHT - ball_radius, ball_y))

        # Update trail
        trail.append((int(ball_x), int(ball_y)))


        # Draw maze
        for cell in grid:
            cell.draw(screen)

        # Draw trail
        for pos in trail:
            pygame.draw.circle(screen, BLUE, pos, 2)

        # Draw start point
        pygame.draw.circle(screen, WHITE, (int(start_x), int(start_y)), ball_radius // 2)

        # Draw target ball
        if not TARGET_REACHED:
            pygame.draw.circle(screen, RED, (int(target_x), int(target_y)), ball_radius)

        # Set player ball color
        if GAME_COMPLETED:
            ball_color = BLUE  # Blue when game completed
        elif collision:
            ball_color = YELLOW  # Yellow on collision
        elif RETURNING_TO_START:
            ball_color = RED  # Red when returning to start
        else:
            ball_color = GREEN  # Green normally
            
        pygame.draw.circle(screen, ball_color, (int(ball_x), int(ball_y)), ball_radius)

        # Draw direction arrow
        arrow_length = ball_radius
        end_x = ball_x + math.cos(ball_angle) * arrow_length
        end_y = ball_y + math.sin(ball_angle) * arrow_length
        pygame.draw.line(screen, WHITE, (ball_x, ball_y), (end_x, end_y), 3)

        # Controls info
        font = pygame.font.SysFont(None, 24)
        controls_text = font.render("Controls: W/S (left), O/L (right), R (reset)", True, WHITE)
        screen.blit(controls_text, (10, 10))

        # Game status
        status_text = ""
        if GAME_COMPLETED:
            status_text = "Rescue sucessfull."
        elif RETURNING_TO_START:
            status_text = "Aquired target, get back to the start."
        else:
            status_text = "Get the red ball."
        
        status_render = font.render(status_text, True, WHITE)
        screen.blit(status_render, (10, 40))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
