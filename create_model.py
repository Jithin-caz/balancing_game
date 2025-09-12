import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import time
import os
import torch
import torch.nn as nn
from typing import Callable

def test_random_agent(env_name, episodes=10):
    """
    Tests an agent that takes random actions in the environment.
    """
    print("--- 🤖 Testing with a Random Agent ---")
    test_env = gym.make(env_name, render_mode='human')
    scores = []
    for episode in range(1, episodes + 1):
        state, info = test_env.reset()
        done = False
        score = 0
        while not done:
            action = test_env.action_space.sample()
            n_state, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            score += reward
        scores.append(score)
        print(f'Episode:{episode} Score:{score:.0f}')
    test_env.close()
    print(f"Average Score: {np.mean(scores):.2f}")
    print("---------------------------------------\n")

def build_optimized_dqn_agent(env):
    """
    Builds an optimized DQN agent for CartPole-v1.
    """
    policy_kwargs = dict(
        net_arch=[64, 64],  # Simpler network for faster learning
        activation_fn=nn.ReLU,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=dict(eps=1e-5)
    )

    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-3,  # Faster learning rate
        buffer_size=100000,  # Reduced buffer size
        learning_starts=5000,  # Delay training to collect experience
        batch_size=256,  # Larger batch for stable gradients
        gamma=0.99,
        train_freq=1,
        target_update_interval=0,  # Use soft updates
        exploration_fraction=0.1,  # Less exploration needed
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,  # Less randomness at end
        tau=0.005,  # Soft target updates
        gradient_steps=1,
        tensorboard_log="./dqn_cartpole_tensorboard/",
        verbose=1
    )
    return model

def test_trained_agent(model, env_name, episodes=10, render=True):
    """
    Evaluates the performance of the trained agent.
    """
    print("\n--- 🧠 Testing the Trained Agent ---")
    
    render_mode = 'human' if render else None
    eval_env = gym.make(env_name, render_mode=render_mode)
    
    scores = []
    for episode in range(1, episodes + 1):
        state, info = eval_env.reset()
        done = False
        score = 0
        steps = 0
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            score += reward
            steps += 1
            if render:
                time.sleep(0.01)
        scores.append(score)
        print(f'Episode:{episode} Score:{score:.0f} Steps:{steps}')
    
    eval_env.close()
    
    print(f"\n📊 Statistics:")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Min Score: {np.min(scores):.0f}")
    print(f"Max Score: {np.max(scores):.0f}")
    print(f"Std Dev: {np.std(scores):.2f}")
    success_rate = (np.array(scores) >= 195).mean() * 100
    print(f"Success Rate (≥195): {success_rate:.1f}%")
    print("----------------------------------------\n")
    
    return scores

def train_with_early_stopping(model, total_timesteps=200000):
    """
    Train with early stopping when the model achieves good performance.
    """
    # Create evaluation environment
    eval_env = Monitor(gym.make('CartPole-v1'))
    
    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=195, 
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=10000,  # Reduced evaluation frequency
        deterministic=True,
        render=False,
        callback_on_new_best=callback_on_best,
        verbose=1
    )
    
    print("--- 🏋️ Training the Optimized Agent with Early Stopping ---")
    model.learn(
        total_timesteps=total_timesteps,  # Reduced to 200,000 (sufficient for CartPole)
        callback=eval_callback,
        progress_bar=True
    )
    
    eval_env.close()
    return model

if __name__ == "__main__":
    ENV_NAME = 'CartPole-v1'
    MODELS_DIR = "models"
    LOGS_DIR = "logs"
    MODEL_PATH = f"{MODELS_DIR}/dqn_cartpole_optimized_groq_qwen"
    
    # Create directories
    for dir_path in [MODELS_DIR, LOGS_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    print("=" * 50)
    print("🎮 CartPole-v1 DQN Training - Optimized Version")
    print("=" * 50)
    print("\nNote: CartPole-v1 is considered 'solved' when")
    print("the average reward is ≥195 over 100 consecutive trials.")
    print("Maximum possible score per episode: 500\n")
    
    # Create training environment
    train_env = gym.make(ENV_NAME)
    
    # Optional: Test with random agent first
    print("\n📝 Baseline Performance (Random Agent):")
    test_random_agent(ENV_NAME, episodes=5)
    
    # Build the optimized model
    agent_model = build_optimized_dqn_agent(train_env)
    
    # Train with early stopping
    agent_model = train_with_early_stopping(agent_model, total_timesteps=200000)
    
    print("\n--- ✅ Training Complete ---\n")
    
    # Save the model
    print(f"--- 💾 Saving model to {MODEL_PATH}.zip ---")
    agent_model.save(MODEL_PATH)
    print("--- ✅ Model Saved ---\n")
    
    # Load and test the model
    print("--- 📥 Loading model from file... ---")
    loaded_model = DQN.load(MODEL_PATH, env=train_env)
    print("--- ✅ Model Loaded ---\n")
    
    # Final evaluation
    print("=" * 50)
    print("🏆 FINAL EVALUATION")
    print("=" * 50)
    
    # Test without rendering for statistics
    print("\n📈 Performance Statistics (20 episodes):")
    scores = test_trained_agent(loaded_model, ENV_NAME, episodes=20, render=False)
    
    # Visual demonstration
    print("\n🎬 Visual Demonstration (5 episodes):")
    test_trained_agent(loaded_model, ENV_NAME, episodes=5, render=True)
    
    # Clean up
    train_env.close()
    
    print("\n✨ Training session complete!")
    print(f"Model saved at: {MODEL_PATH}.zip")