import pygame
from Maze import *
from Kalman_filter import *
from Utils import *
from Config import *
from Map import *

grid = generate_maze()
kf = KalmanFilter(initial_state=[1, 1, 1], grid=grid, screen=screen)
ogm = OccupancyGridMap(rows=ROWS, cols=COLS,grid = grid)


def main():
    global ball_x, ball_y, ball_angle

    clock = pygame.time.Clock()
    running = True
    target_x, target_y = place_target_randomly(grid, start_x, start_y, ball_radius, CELL_SIZE, WALL_THICKNESS)
    state = {'left_speed': 0, 'right_speed': 0, 'max_speed': wheel_max_speed, 'reset': False}
    trail = []

    while running:
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            handle_keyboard_input(event, state)

        v = (state['right_speed'] + state['left_speed']) / 2
        omega = (state['right_speed'] - state['left_speed']) / wheel_base
        ball_angle += omega

        ball_x, ball_y = update_robot_position(ball_x, ball_y, ball_angle, v, omega, ball_radius, grid, CELL_SIZE, WALL_THICKNESS)
        ball_x = max(ball_radius, min(WIDTH - ball_radius, ball_x))
        ball_y = max(ball_radius, min(HEIGHT - ball_radius, ball_y))

        trail.append((int(ball_x), int(ball_y)))

        update_game_status(ball_x, ball_y, target_x, target_y)
        draw_maze_elements(grid, screen)
        draw_trail(trail, screen)
        pygame.draw.circle(screen, WHITE, (int(start_x), int(start_y)), ball_radius // 2)
        draw_target(target_x, target_y)

        ball_color = determine_ball_color()
        draw_robot(ball_x, ball_y, ball_angle, ball_radius, ball_color, screen)

        sensor_lengths = calculate_sensor_object_distances(ball_x, ball_y, ball_radius, grid, sensor_angles, CELL_SIZE, WALL_THICKNESS)
        draw_sensor_lines(ball_x, ball_y, ball_radius, sensor_angles, sensor_lengths, screen, font_sensors)

        kf.predict(ball_x, ball_y, ball_angle, dt=0.1)
        features = kf.get_observed_features(ball_x, ball_y, ball_angle)
        kf.correct(features)

        draw_ui_texts(screen, state, ball_x, ball_y)

        # nach draw_sensor_lines():
        ogm.update(ball_x, ball_y, sensor_angles, sensor_lengths,ball_radius)
        ogm.draw(screen)


        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
    grid = generate_maze()
    evolve_population(grid)

#TODO: set timer
#TODO: create random obstacle function
#TODO: have ball change color when collision


#NEAT implementation into simulation.py
import neat
from Maze import *
from Kalman_filter import *
from Utils import *
from Config import *
from Map import *

def evaluate_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = run_simulation(net)
    return fitness

def run_simulation(net):
    grid = generate_maze()
    ogm = OccupancyGridMap(rows=ROWS, cols=COLS, grid=grid)
    kf = KalmanFilter(initial_state=[1, 1, 1], grid=grid, screen=screen)

    clock = pygame.time.Clock()
    running = True
    max_steps = 1000
    score = 0

    target_x, target_y = place_target_randomly(grid, start_x, start_y, ball_radius, CELL_SIZE, WALL_THICKNESS)
    ball_x, ball_y, ball_angle = start_x, start_y, 0

    for step in range(max_steps):
        sensor_lengths = calculate_sensor_object_distances(ball_x, ball_y, ball_radius, grid, sensor_angles, CELL_SIZE, WALL_THICKNESS)

        # Normalize sensor distances
        max_dist = math.hypot(WIDTH, HEIGHT)
        inputs = [min(d / max_dist, 1.0) for d in sensor_lengths[:4]]

        output = net.activate(inputs)
        left_speed, right_speed = output[0] * wheel_max_speed, output[1] * wheel_max_speed

        v = (right_speed + left_speed) / 2
        omega = (right_speed - left_speed) / wheel_base
        ball_angle += omega
        ball_x, ball_y = update_robot_position(ball_x, ball_y, ball_angle, v, omega, ball_radius, grid, CELL_SIZE, WALL_THICKNESS)

        if reached_target(ball_x, ball_y, target_x, target_y):
            score += 100
            break

        score += 1  # small reward for survival

        clock.tick(30)

    return score