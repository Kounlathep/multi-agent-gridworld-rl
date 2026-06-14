
import random


def manhattan_distance(pos1, pos2):
    """
    Used for spawn separation in grid-world (4-direction movement only).
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def spawn_positions(grid_size=5):
    """
   Randomly place the cat, mouse, and cheese on the grid
   Distance rules:

    - Mouse-Cheese ≥ 2 steps
    - Cat-Mouse ≥ 3 steps
    """
    while True:
        all_cells = [
            (r, c)
            for r in range(grid_size)
            for c in range(grid_size)
        ]

        jerry, tom, cheese = random.sample(all_cells, 3)

        if (manhattan_distance(jerry, cheese) >= 2 and
                manhattan_distance(jerry, tom) >= 3):
            return jerry, tom, cheese
