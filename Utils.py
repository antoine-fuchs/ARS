import pygame
import math
import random
from Config import *


# Circle-Circle collision detection
def circle_circle_collision(x1, y1, r1, x2, y2, r2):
    distance_squared = (x1 - x2) ** 2 + (y1 - y2) ** 2
    return distance_squared <= ((r1 + r2) ** 2)

#should we here put a function that randomly puts obstacles (rectangles/circles) in the maze?

def check_wall_collision(circle_x, circle_y, r, grid):
    sides = ['top','right','bottom','left']
    for cell in grid:
        x, y = cell.x, cell.y
        walls = [
            pygame.Rect(x, y, CELL_SIZE, WALL_THICKNESS),                    # top
            pygame.Rect(x+CELL_SIZE-WALL_THICKNESS, y, WALL_THICKNESS, CELL_SIZE),  # right
            pygame.Rect(x, y+CELL_SIZE-WALL_THICKNESS, CELL_SIZE, WALL_THICKNESS),  # bottom
            pygame.Rect(x, y, WALL_THICKNESS, CELL_SIZE)                    # left
        ]
        for idx, rect in enumerate(walls):
            if not cell.walls[idx]:
                continue

            cx = max(rect.x, min(circle_x, rect.x + rect.width))
            cy = max(rect.y, min(circle_y, rect.y + rect.height))
            dx, dy = circle_x - cx, circle_y - cy
            if dx*dx + dy*dy <= r*r:
                return rect, sides[idx]
    return None, None


def adjust_ball_position(ball_x, ball_y, r, rect, side):
    if rect is None:
        return ball_x, ball_y
    if   side == 'top':    return ball_x, rect.bottom + r
    elif side == 'bottom': return ball_x, rect.top    - r
    elif side == 'left':   return rect.right + r, ball_y
    elif side == 'right':  return rect.left  - r, ball_y
    return ball_x, ball_y


def cast_sensor(ball_x, ball_y, ball_radius, angle_deg, grid,
                max_range, step=1):
    angle_rad = math.radians(angle_deg)
    # Wir beginnen bei dist = ball_radius
    for dist in range(int(ball_radius), int(max_range), step):
        probe_x = ball_x + dist * math.cos(angle_rad)
        probe_y = ball_y + dist * math.sin(angle_rad)
        # r=0, weil wir nur Punkt prüfen
        collision_rect, _ = check_wall_collision(probe_x, probe_y, 0, grid)
        if collision_rect:
            # Zieh den Radius ab, damit 0 = Berührung
            return dist - ball_radius

    # Keine Wand im Sichtbereich
    return max_range - ball_radius


def calculate_sensor_object_distances(ball_x, ball_y, ball_radius, grid,
                                      sensor_angles, max_range, step=1):
    distances = []
    for angle in sensor_angles:
        d = cast_sensor(ball_x, ball_y, ball_radius, angle,
                        grid, max_range, step)
        distances.append(d)
    return distances


# Function to place target randomly in a valid position
def place_target_randomly(grid, player_x, player_y, ball_radius, CELL_SIZE, WALL_THICKNESS):
    while True:
        # Choose a random cell
        random_cell = random.choice(grid)
        # Calculate center of the cell
        cell_center_x = (random_cell.x) + (CELL_SIZE // 2)
        cell_center_y = (random_cell.y) + (CELL_SIZE // 2)
        
        # Check if it's not too close to the player's starting position
        min_distance = 5 * CELL_SIZE  # Ensure target is at least 5 cells away
        if ((cell_center_x - player_x) ** 2 + (cell_center_y - player_y) ** 2) >= (min_distance ** 2): #so it's at least 25 away??
            # Check if it doesn't collide with walls
            if not check_wall_collision(cell_center_x, cell_center_y, ball_radius, grid)[0]:
                return cell_center_x, cell_center_y
            
def handle_keyboard_input(event, state):
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_w:
            state['left_speed'] = state['max_speed']
        elif event.key == pygame.K_s:
            state['left_speed'] = -state['max_speed']
        elif event.key == pygame.K_o:
            state['right_speed'] = state['max_speed']
        elif event.key == pygame.K_l:
            state['right_speed'] = -state['max_speed']
        elif event.key == pygame.K_r:
            state['reset'] = True

    elif event.type == pygame.KEYUP:
        if event.key in (pygame.K_w, pygame.K_s):
            state['left_speed'] = 0
        if event.key in (pygame.K_o, pygame.K_l):
            state['right_speed'] = 0

def update_robot_position(ball_x, ball_y, ball_angle, v, omega,
                          radius, grid, CELL_SIZE, WALL_THICKNESS):

    dx = v * math.cos(ball_angle)
    dy = v * math.sin(ball_angle)


    steps    = int(max(abs(dx), abs(dy)) / 0.1) + 1
    step_dx  = dx / steps
    step_dy  = dy / steps


    for _ in range(steps):
        test_x = ball_x + step_dx
        rect, side = check_wall_collision(test_x, ball_y, radius, grid)
        if rect is None:
            ball_x = test_x
        else:

            ball_x, ball_y = adjust_ball_position(test_x, ball_y, radius, rect, side)
            break


    for _ in range(steps):
        test_y = ball_y + step_dy
        rect, side = check_wall_collision(ball_x, test_y, radius, grid)
        if rect is None:
            ball_y = test_y
        else:
            ball_x, ball_y = adjust_ball_position(ball_x, test_y, radius, rect, side)
            break

    return ball_x, ball_y


def draw_trail(trail, screen):
    if len(trail) > 2:
        pygame.draw.lines(screen, BLUE, False, trail, 2)


def draw_maze_elements(grid, screen):
    for cell in grid:
        cell.draw(screen)
        if cell.marker:
            pygame.draw.circle(screen, (0, 255, 0), (cell.x, cell.y), 5)
            id_text = font_maze_id.render(str(cell.cell_id), True, WHITE)
            text_rect = id_text.get_rect(center=(cell.x, cell.y))
            screen.blit(id_text, text_rect)


def draw_sensor_lines(ball_x, ball_y, radius, sensor_angles, sensor_lengths, screen, font):
    #font_sensors = pygame.font.SysFont(None, 20)
    for i, angle in enumerate(sensor_angles):
        angle_rad = math.radians(angle)
        start_x = ball_x + radius * math.cos(angle_rad)
        start_y = ball_y + radius * math.sin(angle_rad)
        end_x = ball_x + (radius + int(sensor_lengths[i])) * math.cos(angle_rad)
        end_y = ball_y + (radius + int(sensor_lengths[i])) * math.sin(angle_rad)

        pygame.draw.line(screen, (255, 0, 0), (start_x, start_y), (end_x, end_y), 2)
        label = font.render(f"{int(sensor_lengths[i])}", True, (255, 0, 0))
        label_rect = label.get_rect(center=(end_x, end_y))
        screen.blit(label, label_rect)

def draw_robot(ball_x, ball_y, angle, radius, color, screen):
    pygame.draw.circle(screen, color, (int(ball_x), int(ball_y)), radius)
    end_x = ball_x + math.cos(angle) * radius
    end_y = ball_y + math.sin(angle) * radius
    pygame.draw.line(screen, WHITE, (ball_x, ball_y), (end_x, end_y), 3)


def draw_ui_texts(screen, state, ball_x, ball_y):
    # Controls info
    controls_text = font_hints.render("Controls: W/S (left), O/L (right), R (reset)", True, WHITE)
    screen.blit(controls_text, (10, 10))

    # Game status
    if GAME_COMPLETED:
        status_text = "Rescue successful."
    elif RETURNING_TO_START:
        status_text = "Acquired target. Get back to the start."
    else:
        status_text = "Get the red ball."

    status_render = font_hints.render(status_text, True, WHITE)
    screen.blit(status_render, (10, 40))

    # Speed info
    speed_text = font_speed.render(f"L:{state['left_speed']:.0f} R:{state['right_speed']:.0f}", True, BLUE)
    text_rect = speed_text.get_rect(center=(ball_x, ball_y))
    screen.blit(speed_text, text_rect)

def update_game_status(ball_x, ball_y, target_x, target_y):
    global TARGET_REACHED, RETURNING_TO_START, GAME_COMPLETED
    if not TARGET_REACHED and circle_circle_collision(ball_x, ball_y, ball_radius, target_x, target_y, ball_radius):
        TARGET_REACHED = True
        RETURNING_TO_START = True
    elif RETURNING_TO_START and circle_circle_collision(ball_x, ball_y, ball_radius, start_x, start_y, ball_radius):
        GAME_COMPLETED = True
        RETURNING_TO_START = False


def draw_target(target_x, target_y):
    if not TARGET_REACHED:
        pygame.draw.circle(screen, RED, (int(target_x), int(target_y)), ball_radius)

def determine_ball_color():
    if GAME_COMPLETED:
        return BLUE
    elif RETURNING_TO_START:
        return RED
    else:
        return GREEN