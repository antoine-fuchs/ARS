import sys
import pygame
import math
import neat
from Maze import *
from Kalman_filter import *
from Utils import *
from Config import *
from Map import *

# Initialize Pygame and create window
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NEAT Maze Simulation")

# Initial map and filters (for manual mode)
grid = generate_maze()
kf = KalmanFilter(initial_state=[1, 1, 1], grid=grid, screen=screen)
ogm = OccupancyGridMap(rows=ROWS, cols=COLS, grid=grid)


def main():
    global ball_x, ball_y, ball_angle

    # Starting position
    ball_x, ball_y, ball_angle = start_x, start_y, 0
    clock = pygame.time.Clock()
    running = True

    # Place the target at a random location, different from the start
    target_x, target_y = place_target_randomly(
        grid, start_x, start_y, ball_radius, CELL_SIZE, WALL_THICKNESS
    )

    # Control state for wheel speeds and reset flag
    state = {'left_speed': 0, 'right_speed': 0, 'max_speed': wheel_max_speed, 'reset': False}
    trail = []

    while running:
        # Handle events so the window doesn't freeze
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            handle_keyboard_input(event, state)

        screen.fill(BLACK)

        # Compute differential drive velocities
        v = (state['right_speed'] + state['left_speed']) / 2
        omega = (state['right_speed'] - state['left_speed']) / wheel_base
        ball_angle += omega

        # Move the robot and enforce boundary limits
        ball_x, ball_y = update_robot_position(
            ball_x, ball_y, ball_angle, v, omega,
            ball_radius, grid, CELL_SIZE, WALL_THICKNESS
        )
        ball_x = max(ball_radius, min(WIDTH - ball_radius, ball_x))
        ball_y = max(ball_radius, min(HEIGHT - ball_radius, ball_y))

        # Record the path for drawing the trail
        trail.append((int(ball_x), int(ball_y)))

        # Update game status and draw all elements\        
        update_game_status(ball_x, ball_y, target_x, target_y)
        draw_maze_elements(grid, screen)
        draw_trail(trail, screen)
        pygame.draw.circle(screen, WHITE, (int(start_x), int(start_y)), ball_radius // 2)
        draw_target(target_x, target_y)

        # Determine and draw the robot with its color
        ball_color = determine_ball_color()
        draw_robot(ball_x, ball_y, ball_angle, ball_radius, ball_color, screen)

        # Measure and display sensor readings
        sensor_lengths = calculate_sensor_object_distances(
            ball_x, ball_y, ball_radius,
            grid, sensor_angles, CELL_SIZE, WALL_THICKNESS
        )
        draw_sensor_lines(
            ball_x, ball_y, ball_radius,
            sensor_angles, sensor_lengths,
            screen, font_sensors
        )

        # Kalman filter prediction and correction
        kf.predict(ball_x, ball_y, ball_angle, dt=0.1)
        features = kf.get_observed_features(ball_x, ball_y, ball_angle)
        kf.correct(features)

        # Draw UI texts (e.g., speed info)
        draw_ui_texts(screen, state, ball_x, ball_y)

        # Update and draw the occupancy grid map
        ogm.update(ball_x, ball_y, sensor_angles, sensor_lengths, ball_radius)
        ogm.draw(screen)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # No rendering during training for faster evaluations
        fitness = run_simulation(net, render=False)
        genome.fitness = fitness



def run_simulation(net, render=False):
    """
    Runs a complete simulation episode using the given feed-forward network.
    If render=True, the simulation is visualized in a Pygame window.
    """
    # Create a new maze, occupancy grid map, and Kalman filter for each genome
    grid = generate_maze()
    ogm_sim = OccupancyGridMap(rows=ROWS, cols=COLS, grid=grid)
    kf_sim = KalmanFilter(initial_state=[1, 1, 1], grid=grid, screen=screen)

    # Initial conditions
    ball_x, ball_y, ball_angle = start_x, start_y, 0
    target_x, target_y = place_target_randomly(
        grid, start_x, start_y, ball_radius, CELL_SIZE, WALL_THICKNESS
    )
    score = 0
    clock = pygame.time.Clock()
    trail = []

    for step in range(100):
        # Pump Pygame events even without rendering to keep it responsive
        pygame.event.pump()

        if render:
            # Handle events to keep the window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        # Sensors measure distances to objects
        distances = calculate_sensor_object_distances(
            ball_x, ball_y, ball_radius,
            grid, sensor_angles, CELL_SIZE, WALL_THICKNESS
        )
        # Normalize sensor inputs
        max_dist = math.hypot(WIDTH, HEIGHT)
        inputs = [min(d / max_dist, 1.0) for d in distances[:4]]

        # Activate the network
        outputs = net.activate(inputs)
        left_speed = outputs[0] * wheel_max_speed
        right_speed = outputs[1] * wheel_max_speed

        # Compute and apply motion
        v = (left_speed + right_speed) / 2
        omega = (right_speed - left_speed) / wheel_base
        ball_angle += omega
        ball_x, ball_y = update_robot_position(
            ball_x, ball_y, ball_angle, v, omega,
            ball_radius, grid, CELL_SIZE, WALL_THICKNESS
        )

        # Check for collision with walls
        if check_wall_collision(ball_x, ball_y, ball_radius, grid, CELL_SIZE, WALL_THICKNESS):
            score -= 5

        # Check if target reached
        if circle_circle_collision(
            ball_x, ball_y, ball_radius,
            target_x, target_y, ball_radius
        ):
            score += 500
            break

        # Increment score and record trail
        score += 1
        trail.append((int(ball_x), int(ball_y)))

        if render:
            # Rendering
            screen.fill(BLACK)
            update_game_status(ball_x, ball_y, target_x, target_y)
            draw_maze_elements(grid, screen)
            draw_trail(trail, screen)
            pygame.draw.circle(screen, WHITE, (int(start_x), int(start_y)), ball_radius // 2)
            draw_target(target_x, target_y)

            ball_color = determine_ball_color()
            draw_robot(ball_x, ball_y, ball_angle, ball_radius, ball_color, screen)

            # Display sensors
            draw_sensor_lines(
                ball_x, ball_y, ball_radius,
                sensor_angles, distances,
                screen, font_sensors
            )

            # Update and draw occupancy grid map
            ogm_sim.update(ball_x, ball_y, sensor_angles, distances, ball_radius)
            ogm_sim.draw(screen)

            pygame.display.flip()
            clock.tick(30)

    # Apply distance-to-target penalty
    remaining_distance = math.hypot(ball_x - target_x, ball_y - target_y)
    score -= remaining_distance

    return score


class VisualizeReporter(neat.reporting.BaseReporter):
    def start_generation(self, generation):
        # Remember which generation is currently running
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        # Visualize the best genome of each generation
        print(f"\n=== Visualization after Generation {self.generation} ===")
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        run_simulation(net, render=True)
        # After closing the window, the process continues automatically


if __name__ == "__main__":
    mode = input("Select mode: [1] Manual, [2] NEAT Training: ")
    if mode.strip() == '1':
        main()
    else:
        # Load NEAT configuration
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            "neat-config.txt"
        )
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(
            neat.Checkpointer(
                generation_interval=5,
                filename_prefix="neat-checkpoint-"
            )
        )
        p.add_reporter(VisualizeReporter())

        # Run evolution (without rendering)
        winner = p.run(eval_genomes, n=10)

        # Save the winning genome
        import pickle
        with open("best_genome.pkl", "wb") as f:
            pickle.dump(winner, f)
        print(f"Best Fitness: {winner.fitness}")

        # Final simulation with visualization
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        print("Starting final simulation with visualization")
        run_simulation(winner_net, render=True)

        pygame.quit()
        sys.exit()
