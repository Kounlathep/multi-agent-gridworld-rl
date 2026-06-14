
from agents.q_agent import QAgent

class JerryAgent(QAgent):
    """
    Jerry-specific Q-learning agent.
    Uses a more exploration-focused configuration.
    """
    def __init__(self):
        super().__init__(
            name          = 'Jerry',
            alpha         = 0.1,
            gamma         = 0.9,
            epsilon       = 1.0,
            epsilon_min   = 0.05,
            epsilon_decay = 0.9995,
        )
