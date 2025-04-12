import pygame
import random

# Screen dimensions and grid size
WIDTH, HEIGHT = 800, 800
COLS, ROWS = 10, 10
CELL_SIZE = WIDTH // COLS

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Cell class
class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.walls = [True, True, True, True]  # Top, Right, Bottom, Left
        self.visited = False
    
    def draw(self, screen):
        x = self.x * CELL_SIZE
        y = self.y * CELL_SIZE
        
        if self.walls[0]:  # Top wall
            pygame.draw.line(screen, WHITE, (x, y), (x + CELL_SIZE, y), 2)
        if self.walls[1]:  # Right wall
            pygame.draw.line(screen, WHITE, (x + CELL_SIZE, y), (x + CELL_SIZE, y + CELL_SIZE), 2)
        if self.walls[2]:  # Bottom wall
            pygame.draw.line(screen, WHITE, (x + CELL_SIZE, y + CELL_SIZE), (x, y + CELL_SIZE), 2)
        if self.walls[3]:  # Left wall
            pygame.draw.line(screen, WHITE, (x, y + CELL_SIZE), (x, y), 2)
    
    def check_neighbors(self, grid):
        neighbors = []
        
        def index(x, y):
            if 0 <= x < COLS and 0 <= y < ROWS:
                return x + y * COLS
            return None
        
        # Directions: top, right, bottom, left
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for i, (dx, dy) in enumerate(directions):
            nx, ny = self.x + dx, self.y + dy
            idx = index(nx, ny)
            if idx is not None and not grid[idx].visited:
                neighbors.append(grid[idx])
        
        return random.choice(neighbors) if neighbors else None

# Remove walls between two cells
def remove_walls(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    if dx == 1:
        a.walls[3] = False
        b.walls[1] = False
    elif dx == -1:
        a.walls[1] = False
        b.walls[3] = False
    if dy == 1:
        a.walls[0] = False
        b.walls[2] = False
    elif dy == -1:
        a.walls[2] = False
        b.walls[0] = False

# Generate maze using DFS algorithm
def generate_maze():
    # Create grid
    grid = [Cell(x, y) for y in range(ROWS) for x in range(COLS)]
    
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

# If this script is run directly, show the maze
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
