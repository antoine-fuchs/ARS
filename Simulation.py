import pygame
import math
import neat
from Maze import *
from Kalman_filter import *
from Utils import *
from Config import *
from Map import *

# Pygame initialisieren und Fenster anlegen
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NEAT-Maze-Simulation")

# Initiale Karte und Filter (für den manuellen Modus)
grid = generate_maze()
kf = KalmanFilter(initial_state=[1, 1, 1], grid=grid, screen=screen)
ogm = OccupancyGridMap(rows=ROWS, cols=COLS, grid=grid)


def main():
    """
    Manuelle Steuerung der Simulation mit Tastatur.
    """
    global ball_x, ball_y, ball_angle

    # Anfangsposition
    ball_x, ball_y, ball_angle = start_x, start_y, 0
    clock = pygame.time.Clock()
    running = True
    target_x, target_y = place_target_randomly(
        grid, start_x, start_y, ball_radius, CELL_SIZE, WALL_THICKNESS
    )
    state = {'left_speed': 0, 'right_speed': 0, 'max_speed': wheel_max_speed, 'reset': False}
    trail = []

    while running:
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            handle_keyboard_input(event, state)

        # Differentialantrieb berechnen
        v = (state['right_speed'] + state['left_speed']) / 2
        omega = (state['right_speed'] - state['left_speed']) / wheel_base
        ball_angle += omega

        # Roboter bewegen
        ball_x, ball_y = update_robot_position(
            ball_x, ball_y, ball_angle, v, omega,
            ball_radius, grid, CELL_SIZE, WALL_THICKNESS
        )
        # Begrenzung an den Rändern
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

        # Sensoren messen und anzeigen
        sensor_lengths = calculate_sensor_object_distances(
            ball_x, ball_y, ball_radius,
            grid, sensor_angles, CELL_SIZE, WALL_THICKNESS
        )
        draw_sensor_lines(
            ball_x, ball_y, ball_radius,
            sensor_angles, sensor_lengths,
            screen, font_sensors
        )

        # Kalman-Filter
        kf.predict(ball_x, ball_y, ball_angle, dt=0.1)
        features = kf.get_observed_features(ball_x, ball_y, ball_angle)
        kf.correct(features)

        draw_ui_texts(screen, state, ball_x, ball_y)

        # OGM aktualisieren und zeichnen
        ogm.update(ball_x, ball_y, sensor_angles, sensor_lengths, ball_radius)
        ogm.draw(screen)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def eval_genomes(genomes, config):
    """
    Batch-Evaluierung: Für jedes Genome eine Simulation durchführen und fitness setzen.
    """
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = run_simulation(net, render=True)  # render=False (kein Live-Rendering im Training)
        genome.fitness = fitness


def run_simulation(net, render=False):
    """
    Führt eine komplette Simulationsepisode mit dem gegebenen Feed-Forward-Netzwerk durch.
    Die Fitness setzt sich zusammen aus überlebten Schritten, Bonus fürs Erreichen und Abzug für Restdistanz zum Ziel.
    Wenn render=True, wird die Simulation in einem Pygame-Fenster visualisiert.
    """
    # Neue Karte, Filter und OGM für jedes Genome
    grid = generate_maze()
    ogm_sim = OccupancyGridMap(rows=ROWS, cols=COLS, grid=grid)
    kf_sim = KalmanFilter(initial_state=[1, 1, 1], grid=grid, screen=screen)

    # Startbedingungen
    ball_x, ball_y, ball_angle = start_x, start_y, 0
    target_x, target_y = place_target_randomly(
        grid, start_x, start_y, ball_radius, CELL_SIZE, WALL_THICKNESS
    )
    score = 0
    clock = pygame.time.Clock()
    trail = []

    for step in range(100):
        # Sensoren messen
        lens = calculate_sensor_object_distances(
            ball_x, ball_y, ball_radius,
            grid, sensor_angles, CELL_SIZE, WALL_THICKNESS
        )
        # Normalisieren
        max_dist = math.hypot(WIDTH, HEIGHT)
        inputs = [min(d / max_dist, 1.0) for d in lens[:4]]

        # Netz aktivieren
        out = net.activate(inputs)
        left_speed, right_speed = out[0] * wheel_max_speed, out[1] * wheel_max_speed

        # Bewegung berechnen und anwenden
        v = (left_speed + right_speed) / 2
        omega = (right_speed - left_speed) / wheel_base
        ball_angle += omega
        ball_x, ball_y = update_robot_position(
            ball_x, ball_y, ball_angle, v, omega,
            ball_radius, grid, CELL_SIZE, WALL_THICKNESS
        )

        # Kollision mit Wand?
        if check_wall_collision(ball_x, ball_y, ball_radius, grid, CELL_SIZE, WALL_THICKNESS):
            score -= 50
            

        # Ziel erreicht?
        if circle_circle_collision(
            ball_x, ball_y, ball_radius,
            target_x, target_y, ball_radius
        ):
            score += 500
            break

        score += 1
        trail.append((int(ball_x), int(ball_y)))

        if render:
            # Darstellung
            screen.fill(BLACK)
            update_game_status(ball_x, ball_y, target_x, target_y)
            draw_maze_elements(grid, screen)
            draw_trail(trail, screen)
            pygame.draw.circle(screen, WHITE, (int(start_x), int(start_y)), ball_radius // 2)
            draw_target(target_x, target_y)

            ball_color = determine_ball_color()
            draw_robot(ball_x, ball_y, ball_angle, ball_radius, ball_color, screen)

            # Sensoren anzeigen
            sensor_lengths = lens
            draw_sensor_lines(
                ball_x, ball_y, ball_radius,
                sensor_angles, sensor_lengths,
                screen, font_sensors
            )

            # OGM zeichnen
            ogm_sim.update(ball_x, ball_y, sensor_angles, sensor_lengths, ball_radius)
            ogm_sim.draw(screen)

            pygame.display.flip()
            clock.tick(30)

    # Distanz zum Ziel als Penalty
    remaining_dist = math.hypot(ball_x - target_x, ball_y - target_y)
    score -= remaining_dist

    return score


if __name__ == "__main__":
    mode = input("Modus wählen: [1] Manuell, [2] NEAT-Training: ")
    if mode.strip() == '1':
        main()
    else:
        # NEAT-Konfiguration laden
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

        # Evolution
        winner = p.run(eval_genomes, n=50)

        # Gewinner speichern
        import pickle
        with open("best_genome.pkl", "wb") as f:
            pickle.dump(winner, f)
        print(f"Beste Fitness: {winner.fitness}")

        # Gewinner simulieren (mit Visualisierung)
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        print(winner_net)
        run_simulation(winner_net, render=True)

        pygame.quit()
