# Project: Training a Rocket League Agent using Deep Reinforcement Learning

**Course:** CSCE 642 - Deep Reinforcement Learning

**Team Members:**
- Andrew Zehrer (930005389)
- Mrunmay Deshmukh (635009164)

## Summary

This project trains an AI agent to play Rocket League using Proximal Policy Optimization (PPO) combined with Prioritized Level Replay (PLR) for curriculum learning. We implement a 5-phase training curriculum that progressively teaches the agent from basic ball-touching to aerial maneuvers and competitive 1v1 self-play. The PLR algorithm automatically focuses training on scenarios where the agent struggles most, using TD-error as a difficulty metric. To prevent scenario ID dependence, we introduce task dropout (50% masking) in later phases, enabling the agent to generalize across all scenarios. Built using [RLGym](https://rlgym.org).

## Training Curriculum

| Phase | Name | Goal | Timesteps |
|-------|------|------|-----------|
| 1 | The Touch Agent | Learn to drive toward and touch the ball | ~500M |
| 2 | The Goal Scorer | Score goals on stationary balls | ~500M |
| 2.5 | The Goal Scorer 2.0 | Handle moving balls, rebounds, positioning | ~500M |
| 3 | The Aerial Agent | Touch balls in the air (low aerials) | ~500M |
| 4 | The Generalist | Task dropout for generalization | ~400M |
| 5 | Self-Play 1v1 | Competitive play against itself | ~200M |

### Key Features

- **Prioritized Level Replay (PLR)**: Automatically focuses training on scenarios where the agent struggles most (TD-error based sampling)
- **Curriculum Learning**: Progressive training from simple to complex scenarios
- **Task Dropout**: 50% scenario ID masking in Phase 4+ for generalization
- **73-Dimensional Observations**: 72 standard RLGym dims + 1 scenario index

## Installation

### Prerequisites

- Python 3.9+

### Setup

```bash
# Clone the repository
git clone https://github.com/andrewzehrer/CSCE-642-Rocket-League-RL
cd CSCE-642-Rocket-League-RL

# Create virtual environment
python3 -m venv rlproj
source rlproj/bin/activate  # Linux/macOS
# OR
.\rlproj\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Training

### Run Training Phases

```bash
# Phase 1: Touch Agent (start from scratch)
python scripts/phase_1.py

# Phase 2: Goal Scorer (loads Phase 1 checkpoint)
python scripts/phase_2.py

# Phase 2.5: Goal Scorer 2.0 (loads Phase 2 checkpoint)
python scripts/phase_2.5.py

# Phase 3: Aerial Agent (loads Phase 2.5 checkpoint)
python scripts/phase_3.py

# Phase 4: Generalist (loads Phase 3 checkpoint)
python scripts/phase_4.py

# Phase 5: Self-Play 1v1 (loads Phase 4 checkpoint)
python scripts/phase_5.py
```

### Training Configuration

- **Network**: 512x512 MLP (policy and critic)
- **PPO**: batch_size=50k, minibatch=50k, epochs=1, entropy=0.01
- **Learning Rate**: 1e-4 (policy and critic)
- **PLR Replay Probability**: 60%

## Visualization

RLViser is used to visualize the trained agent. Setup differs by platform.

### Windows

1. Download `rlviser.exe` from [RLViser Releases](https://github.com/VirxEC/rlviser/releases)
2. Place `rlviser.exe` in the `scripts/` folder
3. Run simulation (rlviser starts automatically):
   ```bash
   python scripts/simulate.py
   ```

### macOS

macOS requires running RLViser from source (Rust required).

1. **Install Rust**: https://www.rust-lang.org/tools/install

2. **Download RLViser source**:
   - Download from: [Google Drive](https://drive.google.com/file/d/1c4U-4vsJz9IB9vVb0TqHCOxu2Dv31Uvs/view?usp=sharing)
   - Extract the zip file

3. **Run visualization** (requires two terminals):

   **Terminal 1 - Start RLViser:**
   ```bash
   cd /path/to/rlviser
   cargo run --release
   ```

   **Terminal 2 - Run simulation:**
   ```bash
   cd /path/to/CSCE-642-Rocket-League-RL
   python scripts/simulate.py
   ```

## Simulation

The simulator supports two modes:

### Scenario Mode (default)
Cycles through all PLR training scenarios in 1v0 mode:
```bash
python scripts/simulate.py --mode scenarios
```

### 1v1 Self-Play Mode
Watch the agent play against itself:
```bash
python scripts/simulate.py --mode 1v1
```

### Options

```bash
python scripts/simulate.py --speed 2              # 2x speed
python scripts/simulate.py --speed 0              # Max speed
python scripts/simulate.py --checkpoint /path     # Custom checkpoint
```

## Project Structure

```
CSCE-642-Rocket-League-RL/
├── README.md
├── requirements.txt
├── scripts/
│   ├── phase_1.py                 # Phase 1: Touch Agent
│   ├── phase_2.py                 # Phase 2: Goal Scorer
│   ├── phase_2.5.py               # Phase 2.5: Goal Scorer 2.0
│   ├── phase_3.py                 # Phase 3: Aerial Agent
│   ├── phase_4.py                 # Phase 4: Generalist
│   ├── phase_5.py                 # Phase 5: Self-Play 1v1
│   ├── simulate.py                # Visualization script
│   ├── plr_utils.py               # PLR implementation
│   ├── plr_learner.py             # PLR-aware PPO learner
│   └── data/checkpoints/          # Saved model checkpoints
└── rocket_league_rl/
    ├── rlgym/                     # RLGym library
    ├── rlgym_ppo/                 # PPO implementation
    └── rlgym_tools/               # Custom reward functions
```

## Acknowledgments

- [RLGym](https://rlgym.org) - Rocket League Gym environment
- [RLViser](https://github.com/VirxEC/rlviser) - Visualization tool
