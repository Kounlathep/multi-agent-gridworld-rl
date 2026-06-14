
from agents.q_agent import QAgent

class TomAgent(QAgent):
    """
    Tom-specific Q-learning agent.
    Uses a configuration tuned for chasing Jerry.
    """
    def __init__(self):
        super().__init__(
            name          = 'Tom',
            alpha         = 0.1,
            gamma         = 0.95,
            epsilon       = 1.0,
            epsilon_min   = 0.05,
            epsilon_decay = 0.9995,
        )
