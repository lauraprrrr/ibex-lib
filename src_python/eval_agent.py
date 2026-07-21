import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from bb_env import BBEnv

# Suppress TensorFlow logging if it conflicts
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 1. Define the Neural Network Architecture exactly as in training
class DQN_MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN_MLP, self).__init__()
        self.red = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.red(x)

def evaluate_agent(model_path, num_episodes, host='127.0.0.1', port=8888):
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Using device: {device}")

    # 2. Instantiate Model and Load Pre-trained Weights
    input_dim = 5  # Features: log_nodos, gap_pct, time_pct, depth, accion_ant
    output_dim = 4 # Actions: 0, 1, 2, 3
    
    model = DQN_MLP(input_dim, output_dim).to(device)
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file '{model_path}' not found!")
        return

    # Load weights safely
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 3. PURE EXPLOITATION: Set network to evaluation mode 
    model.eval()
    print(f"[EVAL] Model '{model_path}' loaded successfully. Exploration (epsilon) is OFF.")

    # Dictionary for better metric logging
    action_names = {
        0: "Best-First",
        1: "LBvUB",
        2: "FeasDiving",
        3: "FeasDivingUB"
    }

    # 4. Initialize Environment
    env = BBEnv(host=host, port=port)
    
    print("\n[EVAL] Ready. Waiting for IBEX C++ binary to connect via TCP...")
    print("-" * 70)

    try:
        for ep in range(1, num_episodes + 1):
            # BBEnv reset waits for the next C++ socket connection
            result = env.reset()
            
            # Handle Gym vs Gymnasium tuple unpacking
            if isinstance(result, tuple):
                state = result[0]
            else:
                state = result

            done = False
            ep_reward = 0.0
            steps = 0
            action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

            # 5. Testing Loop (Episode)
            while not done:
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                # Disable gradient calculation for faster/memory-efficient inference
                with torch.no_grad():
                    q_values = model(state_tensor)
                    # Greedy action selection: argmax(Q-values)
                    action = torch.argmax(q_values).item()

                # Execute action
                step_result = env.step(action)
                
                # Handle Gym vs Gymnasium tuple unpacking
                if len(step_result) == 5:
                    next_state, reward, done, truncated, info = step_result
                    done = done or truncated
                else:
                    next_state, reward, done, info = step_result

                # Update metrics
                action_counts[action] += 1
                ep_reward += reward
                steps += 1
                state = next_state

            # Format action distribution for logging
            actions_str = ", ".join([f"{action_names[a]}: {action_counts[a]}" for a in range(4)])
            
            print(f"Problem/Ep {ep:03d} | Total Reward: {ep_reward:7.2f} | Macro-Steps: {steps:4d} | Usage: [{actions_str}]")

    except KeyboardInterrupt:
        print("\n[EVAL] Evaluation interrupted by user. Shutting down gracefully...")
    finally:
        # 6. Clean Exit
        env.close()
        print("[EVAL] Environment closed. Socket resources freed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN Agent on IBEX B&B.")
    parser.add_argument("--model", type=str, default="modelo_dqn_final.pth", help="Path to the saved .pth model file.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of benchmark problems/episodes to evaluate.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="TCP Host IP.")
    parser.add_argument("--port", type=int, default=8888, help="TCP Port.")
    
    args = parser.parse_args()
    
    evaluate_agent(model_path=args.model, num_episodes=args.episodes, host=args.host, port=args.port)