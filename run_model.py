# dqn_cartpole_optimized_groq_qwen.zip and best_model.zip is the best model
import gymnasium as gym
from stable_baselines3 import DQN
import time
import os

def run_simulation(model, env, episodes=50):
    """
    Visualizes the performance of a trained agent in a given environment.

    :param model: The trained Stable-Baselines3 model.
    :param env: The Gymnasium environment to test the agent in.
    :param episodes: The number of episodes to run.
    """
    print("\n--- 🧠 Watching the Trained Agent Play ---")
    for episode in range(1, episodes + 1):
        state, info = env.reset()
        done = False
        score = 0
        while not done:
            # Use the trained model to predict the best action (deterministic=True means no random exploration)
            action, _ = model.predict(state, deterministic=True)
            
            # Perform the action in the environment
            state, reward, terminated, truncated, info = env.step(action)
            
            # An episode is over if the game is terminated or truncated
            done = terminated or truncated
            score += reward
            
            # A small delay to make the visualization easier to follow
            time.sleep(0.01)

        print(f'Episode:{episode} Score:{score}')
        
    env.close()
    print("----------------------------------------\n")


# --- Main execution block ---
if __name__ == "__main__":
    ENV_NAME = 'CartPole-v1'
    MODELS_DIR = "models"
    # The path to the saved model (Stable-Baselines3 automatically adds the .zip)
    MODEL_PATH = f"{MODELS_DIR}/best_model"

    # --- Step 1: Check if the trained model file exists ---
    if not os.path.exists(f"{MODEL_PATH}.zip"):
        print(f"Error: Model not found at {MODEL_PATH}.zip")
        print("Please run the 'train_modern_agent.py' script first to train and save the model.")
    else:
        print(f"--- 📥 Loading model from {MODEL_PATH}.zip ---")
        
        # --- Step 2: Create the environment for visualization ---
        # render_mode='human' will create a pop-up window to show the game
        eval_env = gym.make(ENV_NAME, render_mode='human')
        
        # --- Step 3: Load the pre-trained model ---
        # The environment is passed to the load function to set up the model's action/observation spaces
        model = DQN.load(MODEL_PATH, env=eval_env)
        
        print("--- ✅ Model Loaded ---")
        
        # --- Step 4: Run the simulation with the loaded model ---
        run_simulation(model, eval_env, episodes=50)
