import pygame
import random
from Config import *

# Cell class
class Cell:
    def __init__(self, i, j, cell_id):
        self.i = i  # column index
        self.j = j  # row index
        self.x = i * CELL_SIZE  # pixel x-position
        self.y = j * CELL_SIZE  # pixel y-position
        self.walls = [True, True, True, True]  # Top, Right, Bottom, Left
        self.visited = False
        self.cell_id = cell_id
        self.marker = random.random() < 0.5  # 20% chance to have a marker
    
    def draw(self, screen):
        if self.walls[0]:  # Top wall
            pygame.draw.line(screen, WHITE, (self.x, self.y), (self.x + CELL_SIZE, self.y), 2)
        if self.walls[1]:  # Right wall
            pygame.draw.line(screen, WHITE, (self.x + CELL_SIZE, self.y), (self.x + CELL_SIZE, self.y + CELL_SIZE), 2)
        if self.walls[2]:  # Bottom wall
            pygame.draw.line(screen, WHITE, (self.x + CELL_SIZE, self.y + CELL_SIZE), (self.x, self.y + CELL_SIZE), 2)
        if self.walls[3]:  # Left wall
            pygame.draw.line(screen, WHITE, (self.x, self.y + CELL_SIZE), (self.x, self.y), 2)



    def check_neighbors(self, grid):
        neighbors = []

        def index(i, j):
            if 0 <= i < COLS and 0 <= j < ROWS:
                return i + j * COLS
            return None

        # Directions: top, right, bottom, left
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for dx, dy in directions:
            ni, nj = self.i + dx, self.j + dy
            idx = index(ni, nj)
            if idx is not None and not grid[idx].visited:
                neighbors.append(grid[idx])

        return random.choice(neighbors) if neighbors else None

# Remove walls between two cells
def remove_walls(a, b):
    dx = a.i - b.i
    dy = a.j - b.j
    if dx == 1:
        a.walls[3] = False  # left
        b.walls[1] = False  # right
    elif dx == -1:
        a.walls[1] = False
        b.walls[3] = False
    if dy == 1:
        a.walls[0] = False  # top
        b.walls[2] = False  # bottom
    elif dy == -1:
        a.walls[2] = False
        b.walls[0] = False

# Generate maze using DFS algorithm
def generate_maze():
    # Create grid with cell IDs
    cell_id = 0
    grid = []
    for j in range(ROWS):
        for i in range(COLS):
            grid.append(Cell(i, j, cell_id))
            cell_id += 1

    current = grid[0]
    stack = []
    current.visited = True

    while True:
        next_cell = current.check_neighbors(grid)
        if next_cell:
            next_cell.visited = True
            stack.append(current)
            remove_walls(current, next_cell)
            current = next_cell
        elif stack:
            current = stack.pop()
        else:
            break

    return grid

# Main loop
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Maze Generator")

    grid = generate_maze()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)

        for cell in grid:
            cell.draw(screen)

        pygame.display.flip()

    pygame.quit()