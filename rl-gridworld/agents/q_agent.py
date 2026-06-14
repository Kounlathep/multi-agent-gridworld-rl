
import numpy as np
import random

class QAgent:
    """
    Generic Q-learning agent used by both Jerry and Tom.
    Agent behavior depends on the state and reward it receives,
    not on the character it represents.
    """
    def __init__(self, name, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.name          = name
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table       = {}

    def get_q_values(self, state):
        """
        Returns Q-values for the given state.
        Unseen states are initialized with default values.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)
        return self.q_table[state]

    def choose_action(self, state):
        """
         Selects an action using the epsilon-greedy strategy.
        """
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return int(np.argmax(self.get_q_values(state)))

    def update(self, state, action, reward, next_state):
        """
        Updates the Q-table using the Q-learning update rule.
        """
        current_q  = self.get_q_values(state)[action]
        max_next_q = np.max(self.get_q_values(next_state))
        td_error   = reward + self.gamma * max_next_q - current_q
        self.q_table[state][action] = current_q + self.alpha * td_error

    def decay_epsilon(self):
        """
        Reduces the exploration rate after each episode.
        """
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    def get_stats(self):
        """
        Returns a summary of the agent's current statistics.
        """
        return {
            'name'         : self.name,
            'epsilon'      : round(self.epsilon, 4),
            'q_table_size' : len(self.q_table),
        }
