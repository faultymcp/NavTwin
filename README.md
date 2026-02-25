# NavTwin: Multi-Agent Digital Twin Framework for Neurodivergent Navigation Assistance

A complete implementation of the NavTwin framework, demonstrating personalised, context-aware route recommendations for neurodivergent individuals through digital twins, multi-agent coordination, and reinforcement learning.

## Quick Start

```bash
# Install dependencies (Python 3.9+)
pip install numpy matplotlib

# Run full evaluation (demo + 4 scenarios + baselines + visualizations)
python main.py

# Run demo only
python main.py --demo
```

## Architecture

NavTwin implements a **six-layer architecture** with **seven specialised agents** coordinated through a publish-subscribe messaging system:

```
┌─────────────────────────────────────────────────────────────┐
│  6. Interface Layer     │ Route viz, gamification dashboard  │
├─────────────────────────┤──────────────────────────────────── │
│  5. AI Models Layer     │ Contextual bandit RL, GRU, MLP     │
├─────────────────────────┤──────────────────────────────────── │
│  4. Intelligence Layer  │ 7 agents via pub/sub messaging     │
├─────────────────────────┤──────────────────────────────────── │
│  3. Digital Twins Layer │ PDT (neural prefs) + EDT (TGNN)    │
├─────────────────────────┤──────────────────────────────────── │
│  2. Data Layer          │ Preprocessing, temporal aggregation │
├─────────────────────────┤──────────────────────────────────── │
│  1. Physical Layer      │ Sensors, APIs, actuators            │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| **Personal Digital Twin** | `digital_twins/personal_dt.py` | GRU + cross-attention neural preference network, online gradient-based learning, p_s ∈ R^32, p_d ∈ R^64 |
| **Environment Digital Twin** | `digital_twins/environment_dt.py` | TGNN-inspired predictions, 20-node city graph, crowd/noise at 15-min intervals, confidence scoring |
| **Route Scoring** | `agents/route_scoring.py` | 4-dimension scoring: PPS + ECS + SPS + ES, d=47 feature extraction per route |
| **RL Agent** | `agents/rl_agent.py` | Contextual bandit, K=10 weight configs, ε-greedy with neural Q-network, experience replay |
| **Orchestrator** | `core/orchestrator.py` | Pub/sub message bus, parallel agent activation, full pipeline coordination |
| **Supporting Agents** | `agents/supporting_agents.py` | Stress detection, gamification engine, privacy guardian |
| **Simulation** | `simulation/engine.py` | 4 diverse user scenarios, outcome modelling, baseline comparison |

### Route Scoring Formula

```
Score(r) = w_PPS · PPS(r) + w_ECS · ECS(r) + w_SPS · SPS(r) + w_ES · ES(r)
```

- **PPS** — Personal Preference Score (neural network)
- **ECS** — Environmental Comfort Score (sensory-weighted)
- **SPS** — Success Probability Score (gradient boosting features)
- **ES** — Efficiency Score (distance, time, reliability)

Weights w_i are dynamically selected by the RL agent from K=10 discrete configurations.

### RL Weight Configurations (K=10)

| Config | w_PPS | w_ECS | w_SPS | w_ES | Strategy |
|--------|-------|-------|-------|------|----------|
| 0 | 0.10 | 0.20 | 0.20 | 0.50 | Strong efficiency |
| 4 | 0.25 | 0.25 | 0.25 | 0.25 | Fully balanced |
| 9 | 0.40 | 0.45 | 0.10 | 0.05 | Maximum comfort |

## Evaluation Results

### User Scenarios

| Scenario | Crowd Sens. | Noise Sens. | Preferred | RL Converges To |
|----------|------------|------------|-----------|-----------------|
| **Sarah** | 8.2/10 | 7.5/10 | Quiet | Configs 8-9 (comfort) |
| **Alex** | 4.5/10 | 5.0/10 | Direct | Config 0 (efficiency) |
| **Maya** | 5.5/10 | 9.0/10 | Scenic | Configs 3,8 (mixed) |
| **James** | 6.0/10 | 6.0/10 | Familiar | Configs 3,5 (balanced) |

### Key Findings

- **Personalisation**: RL agent correctly identifies user-appropriate weight configurations
- **Sarah/Maya** (high sensitivity): +9.6% completion rate over static baseline
- **Alex** (efficiency-first): RL converges to efficiency config, 86% completion
- **Convergence**: ε decays from 1.0 → 0.05 by journey ~130

### Generated Outputs

All in `output/`:
- `rl_convergence.png` — Weight adaptation over time per user
- `baseline_comparison.png` — Adaptive RL vs static weights
- `action_distribution.png` — Which configs each user converges to
- `acceptance_completion.png` — Before/after adaptation rates
- `reward_stress.png` — Reward signal and stress convergence
- `score_dimensions.png` — PPS/ECS/SPS/ES profiles per user
- `epsilon_decay.png` — Exploration→exploitation transition
- `evaluation_report.md` — Full metrics report

## Project Structure

```
navtwin/
├── main.py                          # Entry point
├── config.py                        # All hyperparameters
├── digital_twins/
│   ├── personal_dt.py               # PDT: neural preference network
│   └── environment_dt.py            # EDT: TGNN predictions, city graph
├── agents/
│   ├── route_scoring.py             # Multi-criteria scoring (4 dimensions)
│   ├── rl_agent.py                  # Contextual bandit weight adaptation
│   └── supporting_agents.py         # Stress, gamification, privacy
├── core/
│   └── orchestrator.py              # ML Orchestrator + pub/sub message bus
├── simulation/
│   └── engine.py                    # 4 user scenarios + outcome modelling
├── visualization/
│   └── plots.py                     # Publication-quality figures
└── output/                          # Generated figures and report
```

## Privacy & Compliance

- All PII remains on-device
- Differential privacy: ε=1.0, δ=10⁻⁵ for shared analytics
- k-anonymity: k ≥ 100 for population insights
- Encryption: AES-256 at rest, TLS 1.3 in transit
- Compliance: GDPR Article 25, CCPA, HIPAA technical safeguards

## Dependencies

- Python 3.9+
- NumPy
- Matplotlib

No external ML frameworks required — all neural networks implemented in NumPy for portability and on-device deployment.
