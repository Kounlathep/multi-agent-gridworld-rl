
from environment.state   import spawn_positions, manhattan_distance
from environment.rewards import (
    JERRY_REACH_CHEESE, JERRY_HIT_CHEESE,
    JERRY_CAUGHT_BY_TOM, JERRY_NORMAL_MOVE, JERRY_DRAW,
    TOM_CATCH_JERRY, TOM_JERRY_REACHED_CHEESE,
    TOM_NORMAL_MOVE, TOM_DRAW, STAY_STILL_PENALTY
)

GRID_SIZE = 5
CHEESE_HP = 5
# 0: up, 1: down, 2: left, 3: right
ACTIONS = {
    0: (-1,  0),
    1: ( 1,  0),
    2: ( 0, -1),
    3: ( 0,  1),
}
NUM_ACTIONS = len(ACTIONS)


class GridWorld:
    """
    GridWorld environment for Tom & Jerry RL.

    Example:
        env = GridWorld()
        state = env.reset()
        state, jr, tr, done = env.step(0, 1)
    """

    def __init__(self, gamma_jerry=0.9, gamma_tom=0.95):
        self.jerry_pos   = None
        self.tom_pos     = None
        self.cheese_pos  = None
        self.cheese_hp   = CHEESE_HP
        self.done        = False
        self.winner      = None
        self.steps       = 0
        self.grid_size   = GRID_SIZE
        self.gamma_jerry = gamma_jerry
        self.gamma_tom   = gamma_tom

    def reset(self):
        """
        Reset the game to start a new episode.
        This will randomly place all characters again,
        clear the game state, and return the initial state.
        """
        self.jerry_pos, self.tom_pos, self.cheese_pos = \
            spawn_positions(self.grid_size)
        self.cheese_hp = CHEESE_HP
        self.done      = False
        self.winner    = None
        self.steps     = 0
        return self.get_jerry_state(), self.get_tom_state()

    def get_jerry_state(self):
        """
        Relative position state for Jerry.
        Represents distances from Jerry to:
          - Cheese
          - Tom
        Also represents cheese hp
        """
        cheese_dr   = self.cheese_pos[0] - self.jerry_pos[0]
        cheese_dc   = self.cheese_pos[1] - self.jerry_pos[1]
        tom_dr      = self.tom_pos[0]    - self.jerry_pos[0]
        tom_dc      = self.tom_pos[1]    - self.jerry_pos[1]
        cheese_hp   = self.cheese_hp
        return (cheese_dr, cheese_dc, tom_dr, tom_dc, cheese_hp)

    def get_tom_state(self):
        """
        Relative position state for Tom.
        Represents distance from Tom to Jerry only.
        """
        jerry_dr    = self.jerry_pos[0] - self.tom_pos[0]
        jerry_dc    = self.jerry_pos[1] - self.tom_pos[1]
        cheese_dr   = self.cheese_pos[0] - self.tom_pos[0]
        cheese_dc   = self.cheese_pos[1] - self.tom_pos[1]
        return (jerry_dr, jerry_dc, cheese_dr, cheese_dc)

    def _move(self, pos, action):
        """
        Moves a position and keeps it inside the grid.
        """
        dr, dc = ACTIONS[action]
        new_row = max(0, min(self.grid_size - 1, pos[0] + dr))
        new_col = max(0, min(self.grid_size - 1, pos[1] + dc))
        return (new_row, new_col)

    def _get_potentials(self):
        """
        Distance rewards and penalties
        """
        dist_t_to_j = manhattan_distance(self.tom_pos, self.jerry_pos)
        dist_j_to_g = manhattan_distance(self.jerry_pos, self.cheese_pos)

        phi_tom   = -2.5 * dist_t_to_j
        phi_jerry = (-2.0 * dist_j_to_g) + (2.0 * dist_t_to_j)

        return phi_jerry, phi_tom

    def step(self, jerry_action, tom_action):
        """
        Executes one step of the game.

        Order of events:
            1. Jerry moves
            2. Check if Jerry reached cheese (win)
            3. Check if Jerry hit Tom (Tom wins)
            4. Tom moves
            5. Distance rewards and penalties
            6. Check if Tom caught Jerry (Tom wins)

        Returns:
            state, jerry_reward, tom_reward, done
        """
        if self.done:
            raise RuntimeError("Episode จบแล้ว กรุณาเรียก reset() ก่อน")

        old_jerry_pos = self.jerry_pos
        old_tom_pos   = self.tom_pos
        old_phi_jerry, old_phi_tom = self._get_potentials()

        self.jerry_pos = self._move(self.jerry_pos, jerry_action)
        self.steps += 1

        jerry_stayed_still = (self.jerry_pos == old_jerry_pos)

        if self.jerry_pos == self.cheese_pos:
            self.cheese_hp -= 1
            if self.cheese_hp == 0:
                jerry_reward = JERRY_REACH_CHEESE
                tom_reward   = TOM_JERRY_REACHED_CHEESE
                self.done    = True
                self.winner  = 'jerry'
                return self.get_jerry_state(), self.get_tom_state(), \
                    jerry_reward, tom_reward, self.done
            else:
                jerry_reward = JERRY_HIT_CHEESE

        if self.jerry_pos == self.tom_pos:
            jerry_reward = JERRY_CAUGHT_BY_TOM
            tom_reward   = TOM_CATCH_JERRY
            self.done    = True
            self.winner  = 'tom'
            return self.get_jerry_state(), self.get_tom_state(), \
                   jerry_reward, tom_reward, self.done

        self.tom_pos = self._move(self.tom_pos, tom_action)
        tom_stayed_still = (self.tom_pos == old_tom_pos)

        if self.tom_pos == self.jerry_pos:
            jerry_reward = JERRY_CAUGHT_BY_TOM
            tom_reward   = TOM_CATCH_JERRY
            self.done    = True
            self.winner  = 'tom'
            return self.get_jerry_state(), self.get_tom_state(), \
                   jerry_reward, tom_reward, self.done

        new_phi_jerry, new_phi_tom = self._get_potentials()

        jerry_reward = JERRY_NORMAL_MOVE + (self.gamma_jerry * new_phi_jerry - old_phi_jerry)
        tom_reward   = TOM_NORMAL_MOVE   + (self.gamma_tom   * new_phi_tom   - old_phi_tom)

        if jerry_stayed_still:
            jerry_reward += STAY_STILL_PENALTY
        if tom_stayed_still:
            tom_reward += STAY_STILL_PENALTY

        return self.get_jerry_state(), self.get_tom_state(), \
               jerry_reward, tom_reward, self.done


    def render(self):
        """
        Displays the grid in the console.

        Symbols:
            M = Jerry
            C = Tom
            G = Cheese
            . = Empty cell

        If multiple objects overlap, Jerry is shown on top.
        """
        grid = [
            ['.' for _ in range(self.grid_size)]
            for _ in range(self.grid_size)
        ]

        grid[self.cheese_pos[0]][self.cheese_pos[1]] = 'G'
        grid[self.tom_pos[0]]  [self.tom_pos[1]]     = 'C'
        grid[self.jerry_pos[0]][self.jerry_pos[1]]   = 'M'

        print(f"\n--- Step {self.steps} ---")
        for row in grid:
            print(' '.join(row))
        print(f"Jerry={self.jerry_pos} | Tom={self.tom_pos} |"
              f"Cheese={self.cheese_pos} | HP={'♥' * self.cheese_hp}")

        if self.done:
            if self.winner == 'jerry':
                print("Jerry Win!")
            else:
                print("Tom Win!")
