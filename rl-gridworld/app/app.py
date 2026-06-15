
import sys
import os
import time
import numpy as np
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from environment.gridworld import GridWorld
from agents.jerry_agent    import JerryAgent
from agents.tom_agent      import TomAgent

st.set_page_config(page_title="Tom & Jerry RL", layout="wide")

st.title("Tom & Jerry: Multi-Agent Reinforcement Learning")

st.sidebar.header("⚙️ Setting")

ai_skill = st.sidebar.selectbox(
    "AI Checkpoint:",
    options=[
        "Episode 100",
        "Episode 1000",
        "Episode 10000",
        "Episode 30000",
        "Episode 50000"
    ]
)
if "100" in ai_skill: ep_selected = 100
elif "1000" in ai_skill: ep_selected = 1000
elif "10000" in ai_skill: ep_selected = 10000
elif "30000" in ai_skill: ep_selected = 30000
else: ep_selected = 50000

speed = st.sidebar.slider("Simulation Speed (seconds per step)", 0.05, 1.0, 0.2)

def load_trained_agents(ep):
    jerry = JerryAgent()
    tom = TomAgent()

    jerry.epsilon = 0.05
    tom.epsilon = 0.05

    jerry_file = os.path.join(project_root, 'models', 'checkpoints', f'jerry_ep_{ep}.npy')
    tom_file   = os.path.join(project_root, 'models', 'checkpoints', f'tom_ep_{ep}.npy')

    if os.path.exists(jerry_file) and os.path.exists(tom_file):
        j_data = np.load(jerry_file, allow_pickle=True).item()
        t_data = np.load(tom_file, allow_pickle=True).item()

        jerry.q_table = {eval(k): v for k, v in j_data.items()}
        tom.q_table   = {eval(k): v for k, v in t_data.items()}
    return jerry, tom

def render_grid_to_emoji(env):
    grid_matrix = [["⬜" for _ in range(5)] for _ in range(5)]

    if env.cheese_pos:
        grid_matrix[env.cheese_pos[0]][env.cheese_pos[1]] = "🧀"
    if env.jerry_pos:
        grid_matrix[env.jerry_pos[0]][env.jerry_pos[1]] = "🐭"
    if env.tom_pos:
        grid_matrix[env.tom_pos[0]][env.tom_pos[1]] = "🐱"

    if env.done and env.winner == 'tom' and env.tom_pos == env.jerry_pos:
        grid_matrix[env.tom_pos[0]][env.tom_pos[1]] = "💥"

    return grid_matrix

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("GridWorld (5×5)")

    start_button = st.button(
        "Start Simulation",
        type="primary",
        use_container_width=True
    )

    grid_placeholder = st.empty()
    status_placeholder = st.empty()

    if start_button:
        env = GridWorld()
        jerry_agent, tom_agent = load_trained_agents(ep_selected)

        jerry_state, tom_state = env.reset()
        env.cheese_hp = 5

        for step in range(1, 201):

            jerry_action = jerry_agent.choose_action(jerry_state)
            tom_action = tom_agent.choose_action(tom_state)

            jerry_state, tom_state, jr, tr, done = env.step(
                jerry_action,
                tom_action
            )

            emoji_grid = render_grid_to_emoji(env)

            grid_html = """
            <div style="
                font-size: 32px;
                line-height: 1.4;
                letter-spacing: 5px;
                text-align: center;
                background-color: #1e1e1e;
                padding: 20px;
                border-radius: 10px;
            ">
            """

            for row in emoji_grid:
                grid_html += " ".join(row) + "<br>"

            grid_html += "</div>"
            grid_placeholder.markdown(grid_html, unsafe_allow_html=True)
            status_placeholder.info(
                f"Step: {step} | Cheese HP: {env.cheese_hp}/5"
            )

            time.sleep(speed)

            if done:
                break

        if env.winner == "jerry":
            st.balloons()
            status_placeholder.success(
                f"Jerry wins in {env.steps} steps"
            )

        elif env.winner == "tom":
            status_placeholder.error(
                f"Tom wins in {env.steps} steps"
            )

        else:
            status_placeholder.warning(
                f"Draw after {env.steps} steps"
            )

with col2:
    st.subheader("Training Results")

    graph_path = os.path.join(project_root, 'assets', 'graphs', 'win_rate_progression.png')

    if os.path.exists(graph_path):
        st.image(
            graph_path,
            caption="Win Rate Progression",
            use_container_width=True
        )
    else:
        st.info("Training graph not available")
