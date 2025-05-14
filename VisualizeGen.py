import pygame
import math
import neat
from Maze import *
from KalmanFilter import *
from Utils import *
from Config import *
from Map import *

class VisualizeReporter(neat.reporting.BaseReporter):
    def __init__(self):
        # 1) Fenster nur einmal öffnen
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Best of Generation Visualization")
        self.clock = pygame.time.Clock()
        self.generation = None
        self.left_wheel_speed = 0
        self.right_wheel_speed = 0

    def start_generation(self, generation):
        global current_generation
        self.generation = generation
        current_generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        print(f"\n=== Generation {self.generation}: Visualizing best genome ===")
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        self._run_visualization(net)

    def _run_visualization(self, net):
        max_steps = 200
        grid = generate_maze()

        vis_kf= KalmanFilter(initial_state=[1, 1, 1], grid=grid, screen=screen)
        est_x, est_y, est_theta = vis_kf.state.flatten()

        # Starte wirklich an den globalen Startkoordinaten
        ball_x, ball_y, ball_angle = start_x, start_y, 0
        target_x, target_y = place_target_randomly(
            grid, ball_x, ball_y, ball_radius, CELL_SIZE, WALL_THICKNESS
        )
        trail = []

        for step in range(max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            distances = calculate_sensor_object_distances(
                ball_x, ball_y, ball_radius,
                grid, sensor_angles, CELL_SIZE, WALL_THICKNESS
            )
            inputs = [d / math.hypot(WIDTH, HEIGHT) for d in distances]

            inputs.extend([
                self.left_wheel_speed  / wheel_max_speed,
                self.right_wheel_speed / wheel_max_speed
            ])

            inputs.extend([
                est_x / WIDTH,
                est_y / HEIGHT
            ])
            outputs = net.activate(inputs)
            alpha = 0.2
            self.left_wheel_speed  = alpha * (outputs[0]*wheel_max_speed) + (1-alpha)*self.left_wheel_speed
            self.right_wheel_speed = alpha * (outputs[1]*wheel_max_speed) + (1-alpha)*self.right_wheel_speed


            #print(f"Step {step}: left: {self.left_wheel_speed}, right: {self.right_wheel_speed}")

            # Bewegung
            v     = (self.left_wheel_speed + self.right_wheel_speed) / 2
            omega = (self.right_wheel_speed - self.left_wheel_speed) / wheel_base
            ball_angle += omega

            # Sub-Stepping + Kollision
            ball_x, ball_y = update_robot_position(
                ball_x, ball_y, ball_angle,
                v, omega,
                ball_radius, grid,
                CELL_SIZE, WALL_THICKNESS
            )
            # Zusätzliche Korrektur (optional)
            rect, side = check_wall_collision(ball_x, ball_y, ball_radius, grid)
            if rect:
                ball_x, ball_y = adjust_ball_position(
                    ball_x, ball_y, ball_radius, rect, side
                )

            trail.append((int(ball_x), int(ball_y)))

            # Rendern auf self.screen
            self.screen.fill(BLACK)
            update_game_status(ball_x, ball_y, target_x, target_y)
            draw_maze_elements(grid, self.screen)
            draw_trail(trail, self.screen)
            # Zeichne Startpunkt an ball_radius//2, wenn du das wirklich möchtest:
            pygame.draw.circle(
                self.screen,
                WHITE,
                (int(start_x), int(start_y)),
                ball_radius // 2
            )
            draw_target(target_x, target_y)

            ball_color = determine_ball_color()
            draw_robot(
                ball_x, ball_y, ball_angle,
                ball_radius, ball_color,
                self.screen
            )
            draw_sensor_lines(
                ball_x, ball_y, ball_radius,
                sensor_angles, distances,
                self.screen, font_sensors
            )

            pygame.display.flip()
            self.clock.tick(30)


    def __getstate__(self):
        state = self.__dict__.copy()
        # drop unpicklable entries
        state.pop('screen', None)
        state.pop('clock', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Fenster/Clock neu anlegen, damit die Visualisierung weiter funktioniert
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()