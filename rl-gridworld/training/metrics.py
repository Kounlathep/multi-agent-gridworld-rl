
class MetricsTracker:
    """
    Tracks training metrics for monitoring and analysis.
    Collected data can be used for progress reporting
    and visualization.
    """
    def __init__(self):
        self.episode_rewards_jerry = []
        self.episode_rewards_tom   = []
        self.episode_lengths       = []
        self.winners               = []
        self.epsilon_history       = []

    def record(self, jerry_total, tom_total, length, winner, epsilon):
        """Records metrics for a single episode"""
        self.episode_rewards_jerry.append(jerry_total)
        self.episode_rewards_tom.append(tom_total)
        self.episode_lengths.append(length)
        self.winners.append(winner)
        self.epsilon_history.append(epsilon)

    def get_winrates(self, last_n=500):
        """
        Computes recent win rates using a sliding window.
        """
        recent = self.winners[-last_n:]
        total  = len(recent)
        if total == 0:
            return 0, 0, 0

        jerry_wr = recent.count('jerry') / total * 100
        tom_wr   = recent.count('tom')   / total * 100
        none_wr  = recent.count('none')  / total * 100
        return round(jerry_wr, 1), round(tom_wr, 1), round(none_wr, 1)

    def summary(self, episode):
        """Prints a summary of the current training progress."""
        jerry_wr, tom_wr, none_wr = self.get_winrates(last_n=500)
        avg_len  = sum(self.episode_lengths[-500:]) / len(self.episode_lengths[-500:])
        avg_jr   = sum(self.episode_rewards_jerry[-500:]) / 500
        avg_tr   = sum(self.episode_rewards_tom[-500:])   / 500

        print(f"Episode {episode:>6} | "
              f"Jerry {jerry_wr:>5.1f}% | "
              f"Tom {tom_wr:>5.1f}% | "
              f"Avg steps {avg_len:>5.1f} | "
              f"ε {self.epsilon_history[-1]:.3f}")
