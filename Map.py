import numpy as np
import pygame
import math
from Config import COLS as DEFAULT_COLS, ROWS as DEFAULT_ROWS, CELL_SIZE, WALL_THICKNESS
from Utils import check_wall_collision

class OccupancyGridMap:
    def __init__(self, grid, rows=DEFAULT_ROWS, cols=DEFAULT_COLS, cell_size=CELL_SIZE, max_range=None):
        # Maze and grid dimensions
        self.grid = grid
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        # Sensor range: diagonal of entire map if not provided
        if max_range is None:
            self.max_range = math.hypot(self.cols * cell_size, self.rows * cell_size)
        else:
            self.max_range = max_range
        # Log-odds arrays for edges
        # Horizontal: (rows+1) x cols
        self.log_h = np.zeros((self.rows + 1, self.cols), dtype=float)
        # Vertical: rows x (cols+1)
        self.log_v = np.zeros((self.rows, self.cols + 1), dtype=float)
        # Inverse sensor model occupancy weight
        self.l_occ = math.log(0.9 / 0.1)
        self.l_free = math.log(0.3 / 0.7)
        # Persistent surface for drawing mapped walls
        width_px = self.cols * cell_size
        height_px = self.rows * cell_size
        self.persist_surf = pygame.Surface((width_px, height_px), flags=pygame.SRCALPHA)
        self.persist_surf.fill((0, 0, 0, 0))
        # List of walls detected in last update
        self.current_walls = []
        # Keep track of already discovered walls for exploration reward
        self.discovered_walls = set()

    def world_to_grid(self, x, y):
        j = min(self.cols - 1, max(0, int(x // self.cell_size)))
        i = min(self.rows - 1, max(0, int(y // self.cell_size)))
        return i, j

    def update(self, robot_x, robot_y, sensor_angles, sensor_distances, ball_radius):
        #updates the log-odds map with current sensor readings & returns an exploration reward (which indicates the number of mewly discovered walls)
        self.current_walls.clear()
        for angle, dist in zip(sensor_angles, sensor_distances):
            if dist >= self.max_range:
                continue

            # Strahl-Endpunkt
            rad = math.radians(angle)
            ex = robot_x + dist * math.cos(rad)
            ey = robot_y + dist * math.sin(rad)

            i_hit, j_hit = self.world_to_grid(ex, ey)

            # 1) free-space updates: unterteile den Strahl in N Schritte
            num_steps = int(dist / (self.cell_size * 0.5))
            for step in range(num_steps):
                t = (step + 0.5) / num_steps
                ix = robot_x + t * (ex - robot_x)
                iy = robot_y + t * (ey - robot_y)
                i_cell, j_cell = self.world_to_grid(ix, iy)
                if (i_cell, j_cell) == (i_hit, j_hit):
                    continue 

                # Horizontal oben/unten
                self.log_h[i_cell    , j_cell] += self.l_free
                self.log_h[i_cell + 1, j_cell] += self.l_free
                # Vertikal links/rechts
                self.log_v[i_cell, j_cell    ] += self.l_free
                self.log_v[i_cell, j_cell + 1] += self.l_free

            # 2) collision check & occupied update
            collision, wall_type = check_wall_collision(ex, ey, ball_radius, self.grid)
            if not collision:
                continue

            i_cell, j_cell = self.world_to_grid(ex, ey)
            if wall_type == 'top'    and self.grid[i_cell*self.cols + j_cell].walls[0]:
                self.log_h[i_cell, j_cell]     += self.l_occ
                self.current_walls.append(('h', i_cell, j_cell))
            elif wall_type == 'bottom' and self.grid[i_cell*self.cols + j_cell].walls[2]:
                self.log_h[i_cell + 1, j_cell] += self.l_occ
                self.current_walls.append(('h', i_cell + 1, j_cell))
            elif wall_type == 'left'   and self.grid[i_cell*self.cols + j_cell].walls[3]:
                self.log_v[i_cell, j_cell]     += self.l_occ
                self.current_walls.append(('v', i_cell, j_cell))
            elif wall_type == 'right'  and self.grid[i_cell*self.cols + j_cell].walls[1]:
                self.log_v[i_cell, j_cell + 1] += self.l_occ
                self.current_walls.append(('v', i_cell, j_cell + 1))
        # clamp log-odds arrays to prevent values from going to extremes
        np.clip(self.log_h, -5.0, 5.0, out=self.log_h)
        np.clip(self.log_v, -5.0, 5.0, out=self.log_v)



        # compute exploration reward
        new_walls = set(self.current_walls) - self.discovered_walls
        exploration_reward = len(new_walls)
        # update discovered walls
        self.discovered_walls.update(new_walls)
        # redraw persistent map after update
        self._draw_persistent()
        return exploration_reward


    # Ensure everything is visualized in the simulation
    def _draw_persistent(self):
        self.persist_surf.fill((0, 0, 0, 0))
        # draw horizontal edges
        for i in range(self.rows + 1):
            for j in range(self.cols):
                val = self.log_h[i, j]
                if val > 0:
                    prob = 1 - 1 / (1 + math.exp(val))
                    alpha = int(200 * prob)
                    x0 = j * self.cell_size
                    y0 = i * self.cell_size
                    pygame.draw.line(
                        self.persist_surf,
                        (150, 0, 0, alpha),
                        (x0, y0), (x0 + self.cell_size, y0), 2
                    )
        # draw vertical edges
        for i in range(self.rows):
            for j in range(self.cols + 1):
                val = self.log_v[i, j]
                if val > 0:
                    prob = 1 - 1 / (1 + math.exp(val))
                    alpha = int(200 * prob)
                    x0 = j * self.cell_size
                    y0 = i * self.cell_size
                    pygame.draw.line(
                        self.persist_surf,
                        (150, 0, 0, alpha),
                        (x0, y0), (x0, y0 + self.cell_size), 2
                    )

    def draw(self, screen):
        screen.blit(self.persist_surf, (0, 0))
        for orient, i, j in self.current_walls:
            x0 = j * self.cell_size
            y0 = i * self.cell_size
            if orient == 'h':
                pygame.draw.line(screen, (255, 0, 0), (x0, y0), (x0 + self.cell_size, y0), 3)
            else:
                pygame.draw.line(screen, (255, 0, 0), (x0, y0), (x0, y0 + self.cell_size), 3)
