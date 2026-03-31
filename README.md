# OpenEnv SME Negotiator: Reinforcement Learning for B2B Contract Negotiation

A rigorous, OpenEnv-compliant reinforcement learning environment for SME B2B contract negotiation. This environment addresses a critical untouched domain identified in the Razorpay "Fix My Itch" dataset: **the asymmetry of negotiation power for Small and Medium Enterprises (SMEs)**.

## Problem Statement

SMEs face severe liquidity crises due to extended payment terms imposed by large corporate buyers. With an "Itch Score" of 82.8 (highest in B2B sectors), the negotiation friction is both massive and current unsolved by existing RL benchmarks.

### Key Regulatory Context
- **MSMED Act (India)**: Corporate buyers must settle payments within 45 days
- **TReDS Platform**: Allows SMEs to auction trade receivables for immediate cash
- **Real-world impact**: Millions of SMEs globally struggle with working capital gaps

## Environment Architecture

### Markov Decision Process (MDP) Formulation

**State Space** ($S$):
- Current opponent proposal (price, days, volume, TReDS willingness)
- Negotiation history (chronological offers and justifications)
- SME constraints (production cost, liquidity threshold, market rates)
- Temporal context (current round, deadline)

**Action Space** ($A$):
- **PROPOSE**: Counter-offer with price/days and natural language justification
- **ACCEPT**: Agree to opponent's current offer
- **REJECT**: Walk away from negotiation

**Reward Function**:
- Intermediate: Always 0.0 (forces long-term credit assignment)
- Terminal: Normalized NPV (0.0-1.0) factoring liquidity penalties and regulatory compliance

### Deterministic Grader

The environment implements a **cheating-proof, reproducible grader** based on closed-form mathematical formulas:

$$Score = \max(0.0, \min(1.0, \frac{U - U_{min}}{U_{max} - U_{min}}))$$

Where:
- $U = NPV_{base} - \Omega(D_{final})$  (final utility)
- $NPV_{base} = Profit \times \left(\frac{1}{(1+r)^{D/365}}\right)$
- $\Omega(D) = \begin{cases} 0 & \text{if } D \le 45 \text{ or } TReDS \\ NPV \times e^{(D-45)/30} - 1 & \text{otherwise} \end{cases}$

## Task Stratification

### Easy Task: Single-Issue Price Optimization
- **Goal**: Maximize unit price while payment terms are fixed at 30 days
- **Complexity**: Simple linear concession strategy
- **Expected Scores**: 0.85-0.95 (most LLMs should succeed)

**Use Case**: Baseline sanity check for environment integration

### Medium Task: Bi-Dimensional Trade-Off with Regulatory Boundaries
- **Goal**: Balance price margin vs payment term reduction
- **Constraint**: SME survival threshold at 60 days; buyer rigid below that
- **Regulatory Limit**: 45-day MSMED Act compliance
- **Expected Scores**: 0.50-0.75 (requires financial reasoning)

**Use Case**: Test Pareto frontier exploration and multi-dimensional optimization

### Hard Task: Non-Linear Financial Restructuring with TReDS
- **Goal**: Overcome rigid buyer constraints via TreDS financial engineering
- **Constraint**: 
  - Buyer absolutely cannot concede below 90 days (treasury lock)
  - SME only survives 30 days
  - Impossible deadline for traditional negotiation
- **Solution Path**: Restructure deal via TReDS, offering strategic price discount
- **Expected Scores**: Near 0.0 for standard models; >0.5 for frontier models

**Use Case**: Test complex multi-hop financial reasoning and legal/regulatory knowledge

## Installation

```bash
git clone https://github.com/scaler/openenv-sme-negotiator.git
cd openenv-sme-negotiator

# Install in development mode
pip install -e .

# Install with dev tools
pip install -e ".[dev]"
```

## Quick Start: Basic Usage

```python
from src.env.sme_negotiation import SMENegotiationEnv
from src.utils.models import NegotiationAction

# Initialize environment
env = SMENegotiationEnv()

# Reset for Easy task with deterministic seed
state = env.reset(task_id="easy", seed=42)

print(f"Initial opponent offer: ₹{state.p_opp}/unit, {state.d_opp} days")
print(f"Your production cost: ₹{state.c_sme}/unit")
print(f"Survival threshold: {state.l_sme} days")

# Generate an action
action = NegotiationAction(
    action_type="PROPOSE",
    proposed_price=95.0,
    proposed_days=30,
    request_treds=False,
    justification="At ₹95/unit with 30-day terms, this contract covers my production costs (₹80/unit) with reasonable margin. Please consider this counter-offer."
)

# Step through environment
observation, reward, terminated, info = env.step(action)

if terminated:
    print(f"Episode ended. Score: {info.get('score', 0.0):.2f}")
else:
    print(f"Buyer counter-offer: ₹{observation.p_opp}/unit, {observation.d_opp} days")
```

## Running the Server

### Local Development
```bash
# Start FastAPI server on http://localhost:8000
python -m uvicorn src.server:app --reload --port 8000
```

### Docker Deployment
```bash
# Build container
docker build -f docker/Dockerfile -t openenv-sme-negotiator:latest .

# Run container
docker run -p 8000:8000 openenv-sme-negotiator:latest
```

### Hugging Face Spaces Deployment
```bash
# Install OpenEnv CLI (when available)
pip install openenv-cli

# Deploy to HF Space
openenv deploy --space-id your-username/sme-negotiator
```

## Advanced: Custom Agent

```python
from src.agents.llm_agent import LLMNegotiationAgent
from src.env.sme_negotiation import SMENegotiationEnv

# Initialize agent and environment
agent = LLMNegotiationAgent(model_name="nemotron-3-super")
env = SMENegotiationEnv()

# Run episode
state = env.reset(task_id="medium", seed=123)
done = False
total_reward = 0.0

while not done:
    # Agent generates action
    action = agent.act(state)
    
    # Environment steps
    state, reward, terminated, info = env.step(action)
    total_reward += reward
    
    if terminated:
        print(f"Final score: {info.get('score', 0.0):.3f}")
        break
```

## Evaluation Methodology

### Phase 1: Automated Validation
- ✅ OpenEnv spec compliance
- ✅ HF Spaces deployment
- ✅ Deterministic grader (no LLM-as-judge)
- ✅ Isolated Docker containerization

### Phase 2: Baseline Performance
- Run Nemotron 3 Super on all tasks (100 episodes per task)
- Verify score variance (non-zero)
- Establish performance envelope

### Phase 3: Human Review
- Real-world economic utility
- Exploit prevention
- Regulatory authenticity

## Scoring Benchmark

Expected baseline (Nemotron 3 Super) performance:
| Task | Easy | Medium | Hard |
|------|------|--------|------|
| Mean Score | 0.88 | 0.62 | 0.08 |
| Pass Rate (score > 0.3) | 100% | 85% | 12% |

(These are illustrative; actual results validate environment quality)

## Key Design Principles

1. **Deterministic & Reproducible**: Fixed seeds guarantee identical trajectories
2. **Secure & Sandboxed**: Server-side grader immune to reward hacking
3. **Realistic Constraints**: Based on actual MSMED Act regulations and TReDS mechanics
4. **Multi-Modal Reasoning**: Requires both quantitative (financial) and qualitative (LLM) capabilities
5. **Scalable**: Supports async rollouts for distributed RL training

## Project Structure

```
openenv-sme-negotiator/
├── src/
│   ├── env/
│   │   └── sme_negotiation.py       # Core MDP environment
│   ├── agents/
│   │   └── llm_agent.py             # Baseline LLM agent
│   ├── utils/
│   │   ├── models.py                # Pydantic schemas
│   │   └── grader.py                # Deterministic grader
│   └── server.py                    # FastAPI WebSocket server
├── docker/
│   └── Dockerfile                   # Container image
├── tests/
│   └── test_environment.py          # Unit tests
├── pyproject.toml                   # Project metadata & dependencies
└── README.md                        # This file
```

## Mathematical Formulation: HardTask TReDS Solution

### Problem Setup
- Buyer demands: ₹95/unit, 120 days
- SME cost: ₹70/unit
- SME survives only: 30 days
- Standard negotiation fails (SME bankruptcy)

### TReDS-Enabled Solution
1. **Agent proposes**: ₹90/unit, 120 days, TReDS=True
2. **Justification**: "By processing via TReDS, your treasury pays the bank in 120 days (satisfying your policy), but I receive discounted funds immediately, solving my liquidity constraint. The ₹5/unit discount (5.3%) covers your platform friction."
3. **Financial Result**:
   - SME Profit: (90-70) × Volume = ₹20/unit value
   - NPV with TReDS: Immediate cash → No delay penalty
   - Score: ~0.65 (significant success for Hard task)

### Why This Is Hard
- Requires understanding TReDS mechanics
- Needs multi-step lookahead (propose → counter → accept)
- Demands natural language persuasion
- Must balance three competing dimensions (price, days, TReDS mode)

## References

1. **MSMED Act**: Section 43B(h), Income Tax Act 1961
2. **TReDS Platform**: RBI Trade Receivables Discounting System
3. **OpenEnv Spec**: https://huggingface.co/docs/openenv
4. **Razorpay "Fix My Itch"**: https://www.razorpay.com/reports/fix-my-itch
5. **Nemotron 3 Super**: NVIDIA's Hybrid Mamba-Transformer model

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{openenv_sme_negotiator_2024,
  title={OpenEnv SME Negotiator: An RL Environment for B2B Contract Negotiation},
  author={Scaler},
  year={2024},
  url={https://github.com/scaler/openenv-sme-negotiator}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear descriptions

## Contact & Support

- **Issues**: GitHub Issues tracker
- **Discussions**: GitHub Discussions
- **Email**: team@scaler.com
