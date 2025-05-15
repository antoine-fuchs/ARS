import sys
import pygame
import math
import neat
from Maze import *
from KalmanFilter import *
from Utils import *
from Config import *
from Map import *
from neat.parallel import ParallelEvaluator
import multiprocessing
from VisualizeGen import VisualizeReporter


# # Initialize Pygame and create window
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NEAT Maze Simulation")



def init_pygame_if_needed(render):
    global screen
    if render and screen is None:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("NEAT Maze Simulation")


def main():
    
    global ball_x, ball_y, ball_angle

    init_pygame_if_needed(render=True)

    MUTATE_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(MUTATE_EVENT, 2000)

    grid = generate_maze()
    kf   = KalmanFilter(initial_state=[start_x, start_y, 0], grid=grid, screen=screen)
    ogm  = OccupancyGridMap(rows=ROWS, cols=COLS, grid=grid)

    # Starting position
    ball_x, ball_y, ball_angle,right_wheel_speed,left_wheel_speed = start_x, start_y, 0,0,0
    clock = pygame.time.Clock()
    running = True

    # Place the target at a random location, different from the start
    target_x, target_y = place_target_randomly(
        grid, start_x, start_y, ball_radius, CELL_SIZE, WALL_THICKNESS
    )

    # Control state for wheel speeds and reset flag
    state = {'left_speed': 0, 'right_speed': 0, 'max_speed': wheel_max_speed, 'reset': False}
    trail = []

    #For rescue timer
    TOTAL_TIME = 180  # seconds
    start_time = pygame.time.get_ticks()

    while running:
        # Handle events so the window doesn't freeze
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == MUTATE_EVENT:
                mutate_maze(grid, changes=5) 
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
        kf.predict(v, omega, dt = 1)  # dt = 1/Frame (30 Hz)
        features = kf.get_observed_features(ball_x, ball_y, ball_angle)
        for feat in features:
            lm_pos = kf.landmark_map[feat['id']]
            kf.correct(feat['measurement'], lm_pos)

        # Draw UI texts (e.g., speed info)
        draw_ui_texts(screen, state, ball_x, ball_y)

        

        # Update and draw the occupancy grid map
        ogm.update(ball_x, ball_y, sensor_angles, sensor_lengths, ball_radius)
        ogm.draw(screen)

        pygame.display.flip()
        clock.tick(40)

        #Make the timer countdown
        now = pygame.time.get_ticks()
        remaining = TOTAL_TIME - (now - start_time) / 1000
        if remaining <= 0:
            print("Time's up!")
            running = False
        else:
            print(f"Remaining time: {remaining:.1f} seconds")

    pygame.quit()
    sys.exit()



def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # No rendering during training for faster evaluations
        fitness = run_simulation(net, render=False)
        genome.fitness = fitness
        return fitness
    
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return run_simulation(net, render=False)



def run_simulation(net, render=False):
    init_pygame_if_needed(render)
    max_steps = 200

    # Karte & Filter für den Simulationslauf
    grid    = generate_maze()
    ogm_sim = OccupancyGridMap(rows=ROWS, cols=COLS, grid=grid)
    kf_sim  = KalmanFilter(initial_state=[start_x, start_y, 0], grid=grid, screen=screen)

    est_x, est_y, est_theta = kf_sim.state.flatten()


    # Initial conditions
    ball_x, ball_y, ball_angle,left_wheel_speed,right_wheel_speed = start_x, start_y, 0,0,0
    target_x, target_y = place_target_randomly(
        grid, start_x, start_y, ball_radius, CELL_SIZE, WALL_THICKNESS
    )
    score = 0
    clock = pygame.time.Clock()
    trail = []

    for step in range(max_steps):
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
        inputs = [min(d / max_dist, 1.0) for d in distances[:12]]

        inputs.extend([
            left_wheel_speed  / wheel_max_speed,
            right_wheel_speed / wheel_max_speed
        ])

        inputs.extend([
            est_x / WIDTH,
            est_y / HEIGHT
        ])


        # Activate the network
        outputs = net.activate(inputs)
        alpha = 0.2 #for smoothing out the wheel speeds
        left_wheel_speed  = alpha * (outputs[0]*wheel_max_speed) + (1-alpha)*left_wheel_speed
        right_wheel_speed = alpha * (outputs[1]*wheel_max_speed) + (1-alpha)*right_wheel_speed


        # Compute and apply motion
        v = (left_wheel_speed + right_wheel_speed) / 2
        omega = (right_wheel_speed - left_wheel_speed) / wheel_base
        ball_angle += omega



        prev_x, prev_y = ball_x, ball_y
        ball_x, ball_y = update_robot_position(
            ball_x, ball_y, ball_angle, v, omega,
            ball_radius, grid, CELL_SIZE, WALL_THICKNESS
        )
        # Update and draw occupancy grid map, get exploration reward


        rect, side = check_wall_collision(ball_x, ball_y, ball_radius, grid)
        if rect:
            score -= 2  # Penalty for hitting a wall
            ball_x, ball_y = adjust_ball_position(
                ball_x, ball_y, ball_radius, rect, side
            )


        trail.append((int(ball_x), int(ball_y)))


    exploration_reward = ogm_sim.update(ball_x, ball_y, sensor_angles, distances, ball_radius)
    print(exploration_reward)

    score += exploration_reward*30

    # Apply distance-to-target penalty
    #remaining_distance = math.hypot(ball_x - target_x, ball_y - target_y)
    #score -= remaining_distance
    distance_from_start = math.hypot(ball_x - start_x, ball_y - start_y)
    print(f"Distance from start: {distance_from_start}")    
    # Optional: skaliere den Reward über einen Faktor
    DISTANCE_REWARD_FACTOR = 5
    score += distance_from_start * DISTANCE_REWARD_FACTOR

    return score



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
        p.add_reporter(neat.StatisticsReporter())
        p.add_reporter(
            neat.Checkpointer(
                generation_interval=5,
                filename_prefix="checkpoints/neat-checkpoint-"
            )
        )
        # Visualisierung erst hier hinzufügen
        p.add_reporter(VisualizeReporter())
        import os
        os.environ["SDL_VIDEODRIVER"] = "dummy"


        pe = ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
        winner = p.run(pe.evaluate, n=500)

        # Save the winning genome
        import pickle
        with open("checkpionts/best_genome.pkl", "wb") as f:
            pickle.dump(winner, f)
        print(f"Best Fitness: {winner.fitness}")

        # Final simulation with visualization
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        print("Starting final simulation with visualization")
        run_simulation(winner_net, render=True)

        pygame.quit()
        sys.exit()
