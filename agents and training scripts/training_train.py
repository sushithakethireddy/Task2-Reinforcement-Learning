"""
Training Script - Partial Observability Variant
Student ID: 16387858 | ID mod 7 = 4
"""
import os
import sys
import json
import asyncio
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.getcwd(), "ChefsHatGYM", "src"))

from rooms.room import Room
from agents_ppo_agent import PPOAgent
from agents_ppo_lstm_agent import PPOLSTMAgent
from agents_random_agent import RandomAgent


def run_room(agent, agent_name, num_matches, output_folder, obs_level):
    """Run one experiment using the real ChefsHat Room."""
    os.makedirs(output_folder, exist_ok=True)

    room = Room(
        run_remote_room=False,
        room_name=agent_name,
        max_matches=num_matches,
        output_folder=output_folder,
        save_game_dataset=True,
        save_logs_game=False,
        save_logs_room=False
    )

    # 3 random opponents
    for i in range(3):
        opp = RandomAgent(
            name=f"Random{i}",
            log_directory=output_folder,
            verbose_console=False
        )
        room.connect_player(opp)

    # Our learning agent
    room.connect_player(agent)

    asyncio.run(room.run())
    return agent


def run_single_experiment(exp_name, agent_type, obs_level, num_matches=100, seed=42, lr=3e-4, hidden_size=128):
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\nExperiment : {exp_name}")
    print(f"Agent      : {agent_type}")
    print(f"Obs Level  : {obs_level}")
    print(f"Matches    : {num_matches}")

    output_folder = os.path.join("results", exp_name)
    os.makedirs(output_folder, exist_ok=True)

    obs_size = 200
    action_size = 200

    if agent_type == "PPO":
        agent = PPOAgent(
            name="PPO_Agent",
            log_directory=output_folder,
            obs_size=obs_size,
            action_size=action_size,
            lr=lr,
            hidden_size=hidden_size,
            obs_level=obs_level
        )
    elif agent_type == "PPO_LSTM":
        agent = PPOLSTMAgent(
            name="PPO_LSTM_Agent",
            log_directory=output_folder,
            obs_size=obs_size,
            action_size=action_size,
            lr=lr,
            hidden_size=hidden_size,
            obs_level=obs_level
        )
    else:
        agent = RandomAgent(name="Random_Agent", log_directory=output_folder)

    agent = run_room(agent, exp_name, num_matches, output_folder, obs_level)

    # Save model and stats
    model_path = os.path.join(output_folder, f"{exp_name}.pt")
    if hasattr(agent, "save"):
        agent.save(model_path)
        print(f"Model saved: {model_path}")

    stats_path = os.path.join(output_folder, f"{exp_name}_stats.json")
    if hasattr(agent, "training_stats"):
        with open(stats_path, "w") as f:
            json.dump(agent.training_stats, f, indent=2)
        print(f"Stats saved: {stats_path}")

    return agent


def run_experiment_1(num_matches=100):
    print("\n### EXPERIMENT 1: Observability Level Comparison ###")
    for level in [1, 2, 3, 4]:
        run_single_experiment(f"exp1_ppo_lstm_level{level}", "PPO_LSTM", level, num_matches)


def run_experiment_2(num_matches=100):
    print("\n### EXPERIMENT 2: PPO vs PPO_LSTM Memory Comparison ###")
    for agent_type in ["PPO", "PPO_LSTM"]:
        run_single_experiment(f"exp2_{agent_type}_level2", agent_type, 2, num_matches)


def run_experiment_3(num_matches=100):
    print("\n### EXPERIMENT 3: Robustness Across Seeds ###")
    for seed in [42, 123, 456]:
        run_single_experiment(f"exp3_ppo_lstm_seed{seed}", "PPO_LSTM", 2, num_matches, seed=seed)


def run_experiment_4(num_matches=100):
    print("\n### EXPERIMENT 4: Hyperparameter Sensitivity ###")
    configs = [
        {"lr": 1e-3,  "hidden_size": 128, "tag": "lr1e3_h128"},
        {"lr": 3e-4,  "hidden_size": 128, "tag": "lr3e4_h128"},
        {"lr": 1e-4,  "hidden_size": 128, "tag": "lr1e4_h128"},
        {"lr": 3e-4,  "hidden_size": 64,  "tag": "lr3e4_h64"},
        {"lr": 3e-4,  "hidden_size": 256, "tag": "lr3e4_h256"},
    ]
    for c in configs:
        run_single_experiment(f"exp4_ppo_lstm_{c['tag']}", "PPO_LSTM", 2, num_matches,
                              lr=c["lr"], hidden_size=c["hidden_size"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="2", choices=["all", "1", "2", "3", "4"])
    parser.add_argument("--matches", type=int, default=100)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    if args.exp == "all" or args.exp == "1":
        run_experiment_1(args.matches)
    if args.exp == "all" or args.exp == "2":
        run_experiment_2(args.matches)
    if args.exp == "all" or args.exp == "3":
        run_experiment_3(args.matches)
    if args.exp == "all" or args.exp == "4":
        run_experiment_4(args.matches)

    print("\nAll experiments complete.")