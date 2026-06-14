
import sys
sys.path.insert(0, '/content/drive/MyDrive/rl-gridworld')

import numpy as np
from environment.gridworld import GridWorld
from agents.jerry_agent    import JerryAgent
from agents.tom_agent      import TomAgent
from training.metrics      import MetricsTracker
from environment.rewards   import JERRY_DRAW, TOM_DRAW

TOTAL_EPISODES = 50_000
MAX_STEPS      = 200
LOG_EVERY      = 2_000
SAVE_EVERY     = [100, 1_000, 5_000, 10_000, 20_000,30_000,40_000,50_000]


def train():
    env     = GridWorld()
    jerry   = JerryAgent()
    tom     = TomAgent()
    metrics = MetricsTracker()
    checkpoints = {}

    print("=" * 60)
    print("Training Tom & Jerry by Q-Learning")
    print(f"Episodes: {TOTAL_EPISODES:,} | Max steps/ep: {MAX_STEPS}")
    print("=" * 60)

    for episode in range(1, TOTAL_EPISODES + 1):

        jerry_state, tom_state = env.reset()
        total_jerry_reward     = 0
        total_tom_reward       = 0

        for step in range(MAX_STEPS):

            jerry_action = jerry.choose_action(jerry_state)
            tom_action   = tom.choose_action(tom_state)

            next_js, next_ts, jr, tr, done = env.step(jerry_action, tom_action)

            jerry.update(jerry_state, jerry_action, jr, next_js)
            tom.update(tom_state,     tom_action,   tr, next_ts)

            total_jerry_reward += jr
            total_tom_reward   += tr
            jerry_state = next_js
            tom_state   = next_ts

            if done:
                break
        if not env.done:
            jerry.update(jerry_state, jerry_action, JERRY_DRAW, jerry_state)
            tom.update(tom_state,     tom_action,   TOM_DRAW,   tom_state)
            total_jerry_reward += JERRY_DRAW
            total_tom_reward   += TOM_DRAW

        jerry.decay_epsilon()
        tom.decay_epsilon()

        winner = env.winner if env.winner else 'none'
        metrics.record(
            jerry_total = total_jerry_reward,
            tom_total   = total_tom_reward,
            length      = env.steps,
            winner      = winner,
            epsilon     = jerry.epsilon,
        )

        if episode in SAVE_EVERY:
            checkpoints[episode] = {
                'jerry' : {k: v.tolist() for k, v in jerry.q_table.items()},
                'tom'   : {k: v.tolist() for k, v in tom.q_table.items()},
            }
            print(f"•Checkpoint saved: episode {episode}")

        if episode % LOG_EVERY == 0:
            metrics.summary(episode)

    print("\n" + "=" * 60)
    print("Trained successfully")
    jerry_wr, tom_wr, none_wr = metrics.get_winrates(last_n=1000)
    print(f"Win rate (last 1,000 episodes):")
    print(f"  Jerry : {jerry_wr}%")
    print(f"  Tom   : {tom_wr}%")
    print(f"  Draw  : {none_wr}%")
    print("=" * 60)

    return jerry, tom, metrics, checkpoints

if __name__ == '__main__':
    jerry, tom, metrics, checkpoints = train()
