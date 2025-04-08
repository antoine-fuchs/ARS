import pygame
import math
from maze import generate_maze, Cell, CELL_SIZE, WIDTH, HEIGHT, BLACK, WHITE

# Initialize Pygame
pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Differential Drive Ball Simulation")

# Colors
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Ball properties
ball_radius = min(CELL_SIZE // 2 - 4, 15)  # Ball must be smaller than a cell
ball_x, ball_y = CELL_SIZE // 2, CELL_SIZE // 2  # Start in the first cell
ball_angle = 0  # Facing direction in radians

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
                return ball_x, y + WALL_THICKNESS + ball_radius
                
        if cell.walls[1] and wall_type == 'right':  # Right wall
            rect = pygame.Rect(x + CELL_SIZE - WALL_THICKNESS, y, WALL_THICKNESS, CELL_SIZE)
            if circle_rect_collision(ball_x, ball_y, ball_radius, rect.x, rect.y, rect.width, rect.height):
                return x + CELL_SIZE - WALL_THICKNESS - ball_radius, ball_y
                
        if cell.walls[2] and wall_type == 'bottom':  # Bottom wall
            rect = pygame.Rect(x, y + CELL_SIZE - WALL_THICKNESS, CELL_SIZE, WALL_THICKNESS)
            if circle_rect_collision(ball_x, ball_y, ball_radius, rect.x, rect.y, rect.width, rect.height):
                return ball_x, y + CELL_SIZE - WALL_THICKNESS - ball_radius
                
        if cell.walls[3] and wall_type == 'left':  # Left wall
            rect = pygame.Rect(x, y, WALL_THICKNESS, CELL_SIZE)
            if circle_rect_collision(ball_x, ball_y, ball_radius, rect.x, rect.y, rect.width, rect.height):
                return x + WALL_THICKNESS + ball_radius, ball_y
    
    return ball_x, ball_y

def main():
    # Generate maze
    grid = generate_maze()
    
    clock = pygame.time.Clock()
    running = True
    collision = False
    
    # Ball variables 
    global left_wheel_speed, right_wheel_speed, ball_x, ball_y, ball_angle
    
    while running:
        screen.fill(BLACK)
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Controls: W/S for left wheel, O/L for right wheel
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    left_wheel_speed = wheel_max_speed
                elif event.key == pygame.K_s:
                    left_wheel_speed = -wheel_max_speed
                elif event.key == pygame.K_o:
                    right_wheel_speed = wheel_max_speed
                elif event.key == pygame.K_l:
                    right_wheel_speed = -wheel_max_speed
                # Additional key R to reset the ball
                elif event.key == pygame.K_r:
                    ball_x, ball_y = CELL_SIZE // 2, CELL_SIZE // 2
                    ball_angle = 0
            
            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_w, pygame.K_s):
                    left_wheel_speed = 0
                if event.key in (pygame.K_o, pygame.K_l):
                    right_wheel_speed = 0
        
        # Differential drive kinematics
        v = (right_wheel_speed + left_wheel_speed) / 2
        omega = (right_wheel_speed - left_wheel_speed) / wheel_base
        
        # Update position
        ball_angle += omega
        dx = v * math.cos(ball_angle)
        dy = v * math.sin(ball_angle)
        
        next_x = ball_x + dx
        next_y = ball_y + dy
        
        # Collision detection
        collision_x, _ = check_wall_collision(next_x, ball_y, ball_radius, grid)
        collision_y, _ = check_wall_collision(ball_x, next_y, ball_radius, grid)
        
        # Update position based on collisions
        if not collision_x:
            ball_x = next_x
        
        if not collision_y:
            ball_y = next_y
        
        # Adjust position for exact wall touching
        ball_x, ball_y = adjust_ball_position(ball_x, ball_y, ball_radius, grid)
        
        # Check for collision after adjustment (for color change)
        collision, _ = check_wall_collision(ball_x, ball_y, ball_radius - 0.1, grid)
        
        # Keep inside window
        ball_x = max(ball_radius, min(WIDTH - ball_radius, ball_x))
        ball_y = max(ball_radius, min(HEIGHT - ball_radius, ball_y))
        
        # Draw maze
        for cell in grid:
            cell.draw(screen)
        
        # Draw ball with direction arrow
        ball_color = YELLOW if collision else GREEN
        pygame.draw.circle(screen, ball_color, (int(ball_x), int(ball_y)), ball_radius)
        
        # Direction arrow
        arrow_length = ball_radius * 1.5
        end_x = ball_x + math.cos(ball_angle) * arrow_length
        end_y = ball_y + math.sin(ball_angle) * arrow_length
        pygame.draw.line(screen, WHITE, (ball_x, ball_y), (end_x, end_y), 3)
        
        # Display control information
        font = pygame.font.SysFont(None, 24)
        text = font.render("Controls: W/S (left wheel), O/L (right wheel), R (reset)", True, WHITE)
        screen.blit(text, (10, 10))
        
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()

if __name__ == "__main__":
    main()
