# Chef's Hat Reinforcement Learning - Partial Observability Variant
## Module: Generative AI and Reinforcement Learning (7043SCN)
Student ID: 16387858
Variant: ID mod 7 = 4 - Partial Observability Variant
University: Coventry University

## Assigned Variant
Partial Observability (ID mod 7 = 4)
This project studies how limiting what an agent can see affects its learning in a competitive card game. Four observability levels are implemented:

Level 1: Full observation (agent sees everything)
Level 2: Opponent hands are hidden
Level 3: Opponent hands and action history are hidden
Level 4: Minimal observation (agent sees own hand only)

The main question is whether an LSTM agent with memory can perform better than a standard PPO agent when information is hidden.

# How to Run
## Step 1: Install dependencies
pip install chefshats
pip install -r requirements.txt
## Step 2: Run experiments
python training_train.py --exp 1 --matches 200
python training_train.py --exp 2 --matches 200
python training_train.py --exp 3 --matches 200
python training_train.py --exp 4 --matches 200
## Step 3: Generate plots
python evaluation_plot_results.py

# Experiment 1: Observability Level Comparison
PPO-LSTM agent is trained at all four observability levels to see how performance changes as more information is hidden.
# Experiment 2: Memory vs No Memory
Standard PPO (no memory) is compared against PPO-LSTM (with memory) under Level 2 partial observability.
# Experiment 3: Robustness Across Random Seeds
PPO-LSTM at Level 2 is run with three different random seeds (42, 123, 456) to check result stability.
# Experiment 4: Hyperparameter Sensitivity
Five combinations of learning rate and hidden size are tested to find the best settings.

## How to Interpret Results
Figure 1: Learning Curves by Observability Level
Shows how reward changes over training for each of the four levels. A higher curve means better performance.
Figure 2: Final Win Rate by Observability Level
Bar chart showing the average win rate in the last 50 episodes for each level. Higher bar means better performance.
Figure 3: PPO vs PPO-LSTM Comparison
If the PPO-LSTM line is higher than the PPO line, memory helps under partial observability. This is the key result.
Figure 4: Seed Robustness
Shows training curves for 3 seeds with a mean and shaded band. A narrow band means stable and reliable results.
Figure 5: Hyperparameter Sensitivity
Bar chart comparing win rates across different learning rate and hidden size settings.
Figure 6: Policy Entropy
Shows how much the agent explores over time. Higher entropy means more exploration.
Figure 7: Actor and Critic Loss
Shows training loss for both agents. Lower and stable values mean the agent is learning well.
Figure 8: Summary
A combined view of all key results in one figure.