# CSCE 642 Rocket League RL Bot

Trained RL bot using PPO with PLR curriculum (Phase 5, 500M steps).

## Files

```
rlbot_export/
├── CSCE642Bot.cfg    # Main bot config (like BroccoliBot.cfg)
├── CSCE642Bot.py     # Bot code with neural network
├── appearance.cfg    # Car loadout
├── discrete.py       # Neural network architecture
├── act.py            # Action lookup table (90 actions)
├── PPO_POLICY.pt     # Trained model weights
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Setup (Windows)

1. Install Python 3.11+ and add to PATH
2. Install dependencies:
   ```cmd
   pip install torch numpy
   ```
3. Install RLBot from rlbot.org
4. In RLBot, click "Add" and select this folder
5. Start a match!

## Model Info

- **Input**: 73 dims (72 obs + 1 scenario_idx=0.0)
- **Output**: 90 discrete actions
- **Architecture**: [512, 512] with ReLU + Softmax
- **Tick skip**: 8 (decides every 8 ticks)

## Scenario ID

The model was trained with `scenario_idx` appended to observations. In real games, this is set to **0.0** (see line 177 in CSCE642Bot.py).
