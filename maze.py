import pygame
import random
from Config import *

# Cell class
class Cell:
    def __init__(self, i, j, cell_id):
        self.i = i
        self.j = j
        self.x = i * CELL_SIZE
        self.y = j * CELL_SIZE
        self.walls = [True, True, True, True]
        self.visited = False
        self.cell_id = cell_id
        self.marker = random.random() < 0.7

    def draw(self, screen):
        if self.walls[0]:  # Top
            pygame.draw.line(screen, WHITE, (self.x, self.y), (self.x + CELL_SIZE, self.y), 2)
        if self.walls[1]:  # Right
            pygame.draw.line(screen, WHITE, (self.x + CELL_SIZE, self.y), (self.x + CELL_SIZE, self.y + CELL_SIZE), 2)
        if self.walls[2]:  # Bottom
            pygame.draw.line(screen, WHITE, (self.x + CELL_SIZE, self.y + CELL_SIZE), (self.x, self.y + CELL_SIZE), 2)
        if self.walls[3]:  # Left
            pygame.draw.line(screen, WHITE, (self.x, self.y + CELL_SIZE), (self.x, self.y), 2)

    def check_neighbors(self, grid):
        neighbors = []
        def index(i, j):
            if 0 <= i < COLS and 0 <= j < ROWS:
                return i + j * COLS
            return None
        dirs = [(0,-1),(1,0),(0,1),(-1,0)]
        for dx, dy in dirs:
            idx = index(self.i+dx, self.j+dy)
            if idx is not None and not grid[idx].visited:
                neighbors.append(grid[idx])
        return random.choice(neighbors) if neighbors else None

# Remove walls between two cells
def remove_walls(a, b):
    dx = a.i - b.i
    dy = a.j - b.j
    if dx == 1:
        a.walls[3] = False; b.walls[1] = False
    elif dx == -1:
        a.walls[1] = False; b.walls[3] = False
    if dy == 1:
        a.walls[0] = False; b.walls[2] = False
    elif dy == -1:
        a.walls[2] = False; b.walls[0] = False

# Add walls between two cells (reverse of remove_walls)
def add_walls(a, b):
    dx = a.i - b.i
    dy = a.j - b.j
    if dx == 1:
        a.walls[3] = True; b.walls[1] = True
    elif dx == -1:
        a.walls[1] = True; b.walls[3] = True
    if dy == 1:
        a.walls[0] = True; b.walls[2] = True
    elif dy == -1:
        a.walls[2] = True; b.walls[0] = True

# Mutate maze: randomly remove or add walls each tick
def mutate_maze(grid, changes=3):
    for _ in range(changes):
        a = random.choice(grid)
        # find neighbors (regardless of visited)
        nbs = []
        for b in grid:
            if abs(a.i - b.i) + abs(a.j - b.j) == 1:
                nbs.append(b)
        if not nbs: continue
        b = random.choice(nbs)
        # toggle
        # if currently walled between, remove; else add
        # detect via a and b's walls
        # pick the wall index
        if a.i - b.i == 1:  # b is left
            if a.walls[3]: remove_walls(a,b)
            else: add_walls(a,b)
        elif a.i - b.i == -1:  # b is right
            if a.walls[1]: remove_walls(a,b)
            else: add_walls(a,b)
        elif a.j - b.j == 1:  # b is top
            if a.walls[0]: remove_walls(a,b)
            else: add_walls(a,b)
        elif a.j - b.j == -1:  # b is bottom
            if a.walls[2]: remove_walls(a,b)
            else: add_walls(a,b)

# Generate maze using DFS algorithm
def generate_maze():
    grid = []
    for j in range(ROWS):
        for i in range(COLS):
            grid.append(Cell(i, j, len(grid)))
    current = grid[0]
    stack = []
    current.visited = True
    while True:
        nxt = current.check_neighbors(grid)
        if nxt:
            nxt.visited = True
            stack.append(current)
            remove_walls(current, nxt)
            current = nxt
        elif stack:
            current = stack.pop()
        else:
            break
    return grid

# Main loop
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Dynamic Maze")

    grid = generate_maze()

    # Setup a custom event for mutation every 2 seconds
    MUTATE_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(MUTATE_EVENT, 2000)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == MUTATE_EVENT:
                mutate_maze(grid, changes=5)

        screen.fill(BLACK)
        for cell in grid:
            cell.draw(screen)

        pygame.display.flip()

    pygame.quit()
