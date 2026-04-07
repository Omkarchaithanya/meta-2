---
title: OpenEnv SME Negotiator
emoji: 🤝
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

<p align="center">
  <img src="logo/OpenEnv-Sme-Negotiator-logo.png" alt="OpenEnv SME Negotiator" width="340">
</p>

<p align="center">
  <strong>Train agents to defend SME liquidity in real-world payment-term negotiations.</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/Omkarchaithanya/sme-negotiator">
    <img src="https://img.shields.io/badge/Hugging%20Face-Space-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face Space">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+">
  </a>
  <a href="https://fastapi.tiangolo.com/">
    <img src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  </a>
  <a href="https://huggingface.co/openenv/">
    <img src="https://img.shields.io/badge/OpenEnv-Compliant-111111?style=for-the-badge" alt="OpenEnv Compliant">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License">
  </a>
</p>

**OpenEnv SME Negotiator** is an OpenEnv-compliant reinforcement learning and agent benchmark for B2B payment-term negotiation. It turns a painfully common SME reality into a reproducible environment: the supplier ships now, the buyer wants 60-90+ day settlement, the SME still has to pay staff, vendors, GST, rent, and debt on time.

If you want an environment where an agent must balance price, payment days, working-capital stress, TReDS, late-payment protection, and dynamic discounting under deterministic grading, this is it.

[GitHub](https://github.com/SkandaGanesha1/ENV) · [Hugging Face Space](https://huggingface.co/spaces/Omkarchaithanya/sme-negotiator) · [OpenEnv Docs](https://huggingface.co/docs/openenv) · [Setup](SETUP.md) · [Evaluation](EVALUATION.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Index](INDEX.md)


</div>
 
---
 
## 📋 Table of Contents
 
- [🚨 The Crisis This Solves](#-the-crisis-this-solves)
- [🏆 Judge Scoring Map](#-judge-scoring-map)
- [💡 Why This Benchmark Is Unique](#-why-this-benchmark-is-unique)
- [🗞️ Live Evidence: News & Official Data](#️-live-evidence-news--official-data)
- [🏗️ Architecture](#️-architecture)
- [🎯 Task Ladder](#-task-ladder)
- [⚙️ Environment Design](#️-environment-design)
- [🚀 Quick Start](#-quick-start)
- [🐍 Python Client](#-python-client)
- [📡 API Surface](#-api-surface)
- [📊 Logging & Evaluation Contract](#-logging--evaluation-contract)
- [🐳 Deployment](#-deployment)
- [🧪 Tests & Validation](#-tests--validation)
- [📖 References](#-references)
 
---
 
## 🚨 The Crisis This Solves
 
<div align="center">
 
<!-- CRISIS STATS PANEL -->
| 📊 Metric | 🔢 Number | 🗓️ Source |
|---|---|---|
| **Delayed MSME receivables (2026 Eco Survey)** | **₹8.1 lakh crore** | Govt of India Economic Survey 2025-26 |
| **Peak delayed payments (2022)** | ₹10.7 lakh crore | GAME-FISME Report 3.0 |
| **Samadhaan portal — total cases filed** | 2,56,892 applications | MSME Samadhaan Portal (Apr 7, 2026) |
| **Amount payable across active cases** | ₹31,255.89 crore | MSME Samadhaan Portal (Apr 7, 2026) |
| **MSMEs in India** | 6.4 crore enterprises | Ministry of MSME |
| **Employment supported** | 32.82 crore people | Economic Survey 2025-26 |
| **Share of India's GDP** | 31.1% | Ministry of MSME |
| **Share of India's exports** | 48.58% | Ministry of MSME |
| **TReDS invoice discounting growth** | ₹0 → ₹2.4 lakh crore | RBI / CEA statement |
| **Razorpay "Fix My Itch" itch score** | 82.8 / 100 | Razorpay B2B Services Survey |
 
</div>

> [!IMPORTANT]
> **The core pain:** A supplier ships goods. The buyer demands 60–90+ day settlement. The SME must still pay staff, vendors, GST, rent, and loan EMIs within 30 days. That cash-flow gap — multiplied across 6.4 crore enterprises — is what this environment makes trainable.
 
> [!NOTE]
> Under **Section 43B(h)** of the Income Tax Act (effective FY 2023-24), buyers who delay MSME payments beyond 45 days lose their tax deduction on that expense. Under **Sections 15–24 of the MSMED Act, 2006**, buyers are liable for **compound interest at 3× the RBI bank rate** on overdue amounts. Despite these protections, ₹8.1 lakh crore remains locked. The negotiation problem is not solved by law alone — it requires agents who can reason about leverage, liquidity, and financing alternatives.

<div align="center">

| Criterion | What We Built | Where to Verify |
|---|---|---|
| **Real-world utility** | Models an active ₹8.1L-crore economic crisis. Would immediately be used by fintech RL labs, MSME policy researchers, and agent benchmark suites. | This README; Economic Survey 2025-26; Razorpay itch score 82.8 |
| **Task & grader quality** | 3 tasks (easy/medium/hard) with deterministic graders in `graders.py`. Scores ∈ [0.0, 1.0]. Hard task resists trivial strategies. | [`sme_negotiator_env/graders.py`](sme_negotiator_env/graders.py); [`sme_negotiator_env/task_config.py`](sme_negotiator_env/task_config.py) |
| **Environment design** | Clean `reset()` → `step()` loop. Rich structured observation (8+ fields). Shaped partial rewards + deterministic terminal reward. Clear episode boundaries. | [`server/sme_environment.py`](server/sme_environment.py); [`sme_negotiator_env/models.py`](sme_negotiator_env/models.py) |
| **Code quality & spec** | `openenv validate` passes. Docker build + run works. HF Space live. Typed Pydantic models throughout. Pytest suite included. | [`openenv.yaml`](openenv.yaml); [`docker/Dockerfile`](docker/Dockerfile); [`tests/`](tests/) |
| **Creativity & novelty** | First OpenEnv environment for B2B payment-term negotiation. Multi-lever reward (price × time × financing × legal clauses). Novel domain for RL/LLM agent research. | Task ladder; `graders.py` reward logic |

</div>

## 💡 Why This Benchmark Is Unique

✅ This benchmark: Survive India's real B2B payment-term crisis with legal, financial,
                   and relational tradeoffs — graded by deterministic code, not vibes.

### Five properties that set this apart
 
<details>
<summary><b>1️⃣ &nbsp;Economic realism — the observation space is a real CFO's dashboard</b></summary>
 
Every field in `NegotiationObservation` maps to a real SME decision variable:
 
| Field | Real-world meaning |
|---|---|
| `buyer_price` | Buyer's opening offer per unit |
| `buyer_days` | Buyer's proposed settlement period |
| `liquidity_threshold_days` | Max days the SME can survive before cash crisis |
| `working_capital_gap` | Shortfall the SME must bridge during the wait |
| `supplier_payment_days` | When the SME must pay its own upstream vendors |
| `interest_rate` | Cost of bridging finance if the SME borrows |
| `buyer_power` | Relative negotiating leverage of the buyer |
| `treds_available` | Whether TReDS discounting is accessible |

An agent that scores well here has genuinely learned to reason about liquidity, not just to pattern-match "say 30 days."
 
</details>

<details>
<summary><b>3️⃣ &nbsp;Progressive difficulty — the hard task genuinely resists frontier models</b></summary>
 
```
EASY   → single lever (compress payment days)
MEDIUM → two levers (days + legal clause)
HARD   → four levers (price × days × TReDS × dynamic discounting NPV)
         in a hostile two-buyer setting with compressed price margin
```
 
On hard mode, a naive "always accept" agent scores ~0. A "always propose 30 days" agent scores ~0. The agent must reason about NPV of early payment vs. discount cost vs. buyer acceptance probability — a genuinely non-trivial sequential decision problem.
 
</details>
 
<details>
<summary><b>4️⃣ &nbsp;Policy-class agnostic — works for RL, LLM, heuristic, and hybrid agents</b></summary>
 
The environment ships with:
- A **typed Python client** (`SMENegotiatorEnv`) for RL and heuristic policies
- A **generic OpenEnv client** (`GenericEnvClient`) for raw dict-based policies
- An **LLM baseline** in `inference.py` targeting any OpenAI-compatible endpoint
- **HTTP + WebSocket APIs** for cross-language agent integrations
- **In-process mode** (`OPENENV_IN_PROCESS=1`) for zero-latency local RL loops
 
</details>
 
<details>
<summary><b>5️⃣ &nbsp;Regulatory anchoring — the environment reflects live Indian law</b></summary>
 
The benchmark is not built on invented rules. Every constraint maps to a real statute or ecosystem mechanism:
 
- **45-day payment window** → MSMED Act, 2006, Sections 15–24
- **Compound interest at 3× RBI rate** → MSMED Act buyer liability
- **TReDS "without recourse" financing** → RBI-defined Trade Receivables Discounting System
- **Late-payment penalty clause** → Section 43B(h), Income Tax Act
- **Dynamic discounting** → Live early-payment platforms (C2FO, M1xchange, RXIL)
 
</details>

---
 
## 🗞️ Live Evidence: News & Official Data
 
> Real-world signals that confirm this is a genuine, unsolved problem — not a classroom exercise.
 
<details>
<summary><b>📰 &nbsp;News Coverage & Market Signal — click to expand</b></summary>
 
<br/>
 
| Source | Headline | Date |
|---|---|---|
| 🏛️ **Economic Survey 2025-26** | *"Delayed payments remain a critical MSME challenge — estimated ₹8.1 lakh crore locked up"* | Jan 29, 2026 |
| 📰 **Business Standard** | *"Delayed payments continue to hit MSMEs, ₹8.1 trillion stuck: Eco Survey"* | Jan 29, 2026 |
| 📰 **Economic Times** | *"Rs 8.1 lakh crore stuck in delayed MSME payments: Economic Survey 2026"* | Jan 29, 2026 |
| 📰 **Financial Express** | *"Delayed payments to MSMEs cross Rs 50,000 crore despite govt push"* | 2024 |
| 🔬 **GAME-FISME-C2FO Report 3.0** | *"Delayed receivables at ₹7.34 lakh crore (Mar 2024), still 4.6% of India's GVA"* | Nov 26, 2025 |
| 🏦 **Razorpay Fix My Itch** | *"Why can't SMEs negotiate favorable payment terms with large buyers?" — itch score 82.8* | 2024 |
| 🏛️ **MSME Samadhaan Portal** | *2,56,892 applications filed; ₹31,255.89 crore payable in active cases* | Apr 7, 2026 |
 
</details>
 
<details>
<summary><b>🏛️ &nbsp;Official Government & Regulatory Links — click to expand</b></summary>
 
<br/>
 
**Legal framework:**
- [MSME Samadhaan delayed-payment portal](https://samadhaan.msme.gov.in/) — Sections 15–24, MSMED Act 2006
- [MSME ODR portal](https://odr.msme.gov.in/) — Online Dispute Resolution for delayed payments
- [Income Tax Section 43B(h)](https://incometaxindia.gov.in/Charts%20%20Tables/Provisions-applicable-to-business-entities.htm) — tax deduction disallowance for late MSME payments
 
**TReDS and financing:**
- [RBI FAQ: Trade Receivables Discounting System](https://www.rbi.org.in/scripts/FAQView.aspx?Id=132) — official "without recourse" definition
- [RBI circular: Expanding TReDS scope](https://www.rbi.org.in/scripts/BS_CircularIndexDisplay.aspx?Id=12510)
- [RBI TReDS statistics](https://www.rbi.org.in/Scripts/TREDSStatisticsView.aspx?TREDSid=40) — ₹0 → ₹2.4L cr growth
 
**Policy reports:**
- [Economic Survey 2025-26, Chapter 8](https://www.indiabudget.gov.in/economicsurvey/doc/eschapter/echap08.pdf)
- [PIB: MSME year-end review with ODR portal](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2209712)
- [PIB: Parliament reply covering Samadhaan, ODR, TReDS](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2153722)
 
</details>
 
<details>
<summary><b>📊 &nbsp;Delayed Payments Timeline — the problem's scale over time</b></summary>
 
<br/>
 
```
Year        Delayed MSME Receivables     Change
─────────────────────────────────────────────────
2022        ₹10.7 lakh crore (peak)      —
2023        ₹ 8.27 lakh crore            ▼ -22.7%
Mar 2024    ₹ 7.34 lakh crore            ▼ -11.2%
Jan 2026    ₹ 8.1  lakh crore (Eco Svy)  ▲ still massive
Apr 2026    ₹31,255 crore in Samadhaan   (active portal cases only)
```
 
**Interpretation:** Despite policy pressure (TReDS, Section 43B(h), Samadhaan), the structural problem — **unequal bargaining power** in B2B payment negotiations — remains unsolved. This is exactly the decision problem this environment benchmarks.
 
</details>

 ## 🎯 Task Ladder
 
<div align="center">
 
```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TASK DIFFICULTY LADDER                           │
├──────────────────┬──────────────────────┬──────────────────────────────┤
│  🟢 EASY         │  🟡 MEDIUM            │  🔴 HARD                     │
│  payment-terms-  │  payment-terms-       │  payment-terms-hard          │
│  easy            │  medium               │                              │
├──────────────────┼──────────────────────┼──────────────────────────────┤
│ Opens: 100 INR   │ Opens: 100 INR        │ Opens: 96 INR / 100 days     │
│        90 days   │        60 days        │ Setting: hostile two-buyer   │
├──────────────────┼──────────────────────┼──────────────────────────────┤
│ Goal: compress   │ Goal: tighten terms   │ Goal: maximize NPV via       │
│ days ≤ 60        │ + add late-payment    │ dynamic discounting          │
│                  │ penalty clause        │ (propose_dynamic_discounting)│
├──────────────────┼──────────────────────┼──────────────────────────────┤
│ Levers: 1        │ Levers: 2             │ Levers: 4                    │
│ (payment days)   │ (days + clause)       │ (price × days × TReDS ×     │
│                  │                       │  discount rate)              │
├──────────────────┼──────────────────────┼──────────────────────────────┤
│ Terminal: 1.0 if │ Terminal: 1.0 if      │ Terminal: NPV-delta score    │
│ days ≤ 60        │ days ≤ 45 + clause    │ vs status quo                │
│ Partial: yes     │ Partial: yes          │ TReDS: changes floor, not    │
│                  │                       │ the terminal score           │
└──────────────────┴──────────────────────┴──────────────────────────────┘
```
 
</div>


> [!WARNING]
> **Hard-mode trap for naive agents:** Setting `use_treds=true` modifies the simulation (lowers the buyer day floor), but the terminal score is **not** earned by TReDS alone. The agent must negotiate dynamic discounting with a coherent `dynamic_discount_annual_rate` to earn hard-task credit. Agents that simply "accept + use_treds" will score close to 0 on the hard task.
 
---
 
## ⚙️ Environment Design
 
### Observation space
 
`NegotiationObservation` — all fields typed via Pydantic:
 
```python
class NegotiationObservation(BaseModel):
    buyer_price: float                  # INR per unit — buyer's current offer
    buyer_days: int                     # settlement days — buyer's current demand
    liquidity_threshold_days: int       # SME's hard limit before cash crisis
    working_capital_gap: float          # INR shortfall to bridge
    supplier_payment_days: int          # when SME must pay its own suppliers
    interest_rate: float                # cost of bridging finance (annual %)
    buyer_power: float                  # 0.0–1.0 buyer negotiating leverage
    treds_available: bool               # TReDS accessible in this episode?
    round_number: int                   # current negotiation round
    max_rounds: int                     # episode length cap
```
 
### Action space
 
`NegotiationAction` — all fields typed, validated on intake:
 
```python
class NegotiationAction(BaseModel):
    action_type: Literal["propose", "accept", "reject"]
    price: float                                  # proposed price in INR/unit
    payment_days: int                             # proposed settlement days
    use_treds: bool = False                       # invoke TReDS financing
    reason: str = ""                              # optional free-text justification
    propose_late_payment_penalty_clause: bool = False   # MEDIUM task lever
    propose_dynamic_discounting: bool = False           # HARD task lever
    dynamic_discount_annual_rate: float = 0.0           # HARD task lever
```
 
> [!NOTE]
> `NegotiationAction` has **no** `message=` field. Use `reason` for any free-text explanation. This is intentional — it forces the agent to express its strategy through structured fields, enabling clean deterministic grading.
 
### Reward shaping
 
- **Partial (step) rewards:** positive signal for reducing `payment_days` gap, penalty for widening it — ensures the agent gets learning signal even in long episodes.
- **Terminal reward:** deterministic score from `graders.py`, always ∈ [0.0, 1.0].
- **Episode boundary:** triggered by `accept`, `reject`, or `max_rounds` exhaustion.
- **No sparse-only rewards:** the agent is never flying blind — every step yields a signal.
 
---

## Why This Exists

For many SMEs, the hardest part of a sale is not winning the order. It is surviving the cash-flow gap after delivery.

Large buyers can push payment cycles far beyond the supplier's comfort zone. The supplier, meanwhile, still needs to pay upstream vendors in roughly 30 days, fund payroll, manage borrowing costs, and preserve the commercial relationship. That creates a negotiation problem that is economically serious, strategically messy, and ideal for agent evaluation.

This repository turns that pain point into a benchmark:

- A deterministic negotiation environment for RL, LLM agents, policy search, and evaluation.
- A typed Python client plus OpenEnv HTTP and WebSocket surfaces.
- Three progressively harder tasks covering term compression, contractual protection, and financing structure.
- A baseline runner that emits audit-friendly `[START]`, `[STEP]`, and `[END]` logs.

## Reality Behind The Benchmark

This project is not built on vague startup folklore. It is anchored in live policy and ecosystem context.

- Razorpay's Fix My Itch lists `Why can't SMEs negotiate favorable payment terms with large buyers?` in B2B Services with an itch score of `82.8`.
- The Ministry of MSME's Samadhaan portal states that delayed-payment cases fall under Sections 15-24 of the MSMED Act, 2006.
- The same official portal states that buyers are liable for compound interest at three times the RBI bank rate if payment is not made within 45 days of acceptance.
- As of April 7, 2026, the official Samadhaan portal showed `2,56,892` applications filed by MSEs and `Rs 31,255.89 crore` in amount payable across total cases filed before MSE Facilitation Councils.
- The Government of India's Economic Survey 2025-26 says delayed payments remain a critical MSME challenge, with an estimated `Rs 8.1 lakh crore` locked up.
- RBI describes TReDS as a financing / discounting system for MSME receivables through multiple financiers, and the receivables are financed on a `without recourse` basis.

That is why this environment does not reduce negotiation to tone or persuasion alone. It forces the agent to reason about liquidity thresholds, price floors, financing options, legal protections, and relationship-preserving compromise.

## Context, News, And Resource Links

### Core problem and official context

- [Razorpay Fix My Itch: Why can't SMEs negotiate favorable payment terms with large buyers?](https://razorpay.com/m/fix-my-itch/)
- [MSME Samadhaan delayed payment portal](https://samadhaan.msme.gov.in/MyMsme/MSEFC/MSEFC_Welcome.aspx)
- [MSME ODR portal](https://odr.msme.gov.in/)
- [MSMED Act delayed-payment context on Samadhaan](https://samadhaan.msme.gov.in/MyMsme/MSEFC/MSEFC_Welcome.aspx)
- [Income Tax summary for Section 43B(h)](https://incometaxindia.gov.in/Charts%20%20Tables/Provisions-applicable-to-business-entities.htm)

### TReDS and financing ecosystem

- [RBI FAQ: Trade Receivables Discounting System (TReDS)](https://www.rbi.org.in/scripts/FAQView.aspx/FAQView.aspx/FAQView.aspx?Id=132)
- [RBI circular: Expanding the scope of TReDS](https://www.rbi.org.in/scripts/BS_CircularIndexDisplay.aspx?Id=12510)
- [RBI TReDS statistics](https://www.rbi.org.in/Scripts/TREDSStatisticsView.aspx?TREDSid=40)
- [TReDS India ecosystem reference](https://www.tredsindia.com/)

### Policy and government updates

- [Economic Survey 2025-26, Chapter 8](https://www.indiabudget.gov.in/economicsurvey/doc/eschapter/echap08.pdf)
- [PIB: MSME year-end review with ODR portal launch note](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2209712)
- [PIB: Parliament reply covering Samadhaan, ODR, and TReDS threshold changes](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2153722)
- [PIB: MSE Scheme on Online Dispute Resolution for Delayed Payments](https://www.pib.gov.in/PressReleasePage.aspx?PRID=1988800&noshow=1)

### News coverage and market signal

- [Economic Times: Rs 8.1 lakh crore stuck in delayed MSME payments](https://economictimes.indiatimes.com/small-biz/sme-sector/rs-8-1-lakh-crore-stuck-in-delayed-msme-payments-eco-survey-2026/articleshow/127765852.cms?from=mdr)
- [Financial Express: Delayed payments to MSMEs cross Rs 50,000 crore despite govt push](https://www.financialexpress.com/business/sme-small-businesses-big-dues-delayed-payments-to-msmes-cross-rs-50000-crore-despite-govt-push-3806557/lite/)

## Why This Is A Strong Agent Benchmark

- **Deterministic grading**: terminal scores come from code in [`sme_negotiator_env/graders.py`](sme_negotiator_env/graders.py), not LLM-as-judge evaluation.
- **Structured economic state**: the observation includes buyer terms, liquidity thresholds, working-capital gap, supplier payment days, interest rate, and buyer power.
- **Real decision tradeoffs**: the agent must decide when to protect price, compress time, invoke TReDS, add a late-payment clause, or propose dynamic discounting.
- **Multi-round behavior**: success depends on sequencing proposals and acceptances, not just producing one plausible answer.
- **Research-friendly interface**: typed models, HTTP API, WebSocket API, Docker support, and in-process execution make it easy to benchmark many policy classes.

## Highlights

- OpenEnv-compatible API on port `7860` with `/health`, `/reset`, `/step`, `/state`, and `/ws`.
- Three tasks: `payment-terms-easy`, `payment-terms-medium`, and `payment-terms-hard`.
- Deterministic terminal graders in [`sme_negotiator_env/graders.py`](sme_negotiator_env/graders.py).
- Typed models for `NegotiationAction`, `NegotiationObservation`, and `NegotiationState`.
- Built-in heuristic fallback plus an OpenAI-compatible LLM baseline in [`inference.py`](inference.py).
- Two execution modes: HTTP/WebSocket with `uv run server`, or fully in-process with `OPENENV_IN_PROCESS=1`.
- Ready for local runs, Docker, and Hugging Face Spaces deployment.

## Quick Start

Runtime: **Python 3.11+**. `uv` is the fastest path.

### Install

```bash
cp .env.example .env
# PowerShell: Copy-Item .env.example .env

uv sync
```

If you also want dev dependencies:

```bash
uv sync --extra dev
```

### Run the environment server

```bash
uv run server
```

The server listens on:

```text
http://127.0.0.1:7860
```

### Run the baseline agent

In another terminal:

```bash
uv run python inference.py
```

### Run without a separate server

```bash
export OPENENV_IN_PROCESS=1
# PowerShell: $env:OPENENV_IN_PROCESS="1"

uv run python inference.py
```

Results are written to `inference_results.json`.

## Python Client

Package name: **`openenv-sme-negotiator`**  
Import path: **`sme_negotiator_env`**  
Typed client: **`SMENegotiatorEnv`**

`NegotiationAction` does **not** have a `message=` field. Use `reason` for free-text justification.

```python
import asyncio
from sme_negotiator_env.client import SMENegotiatorEnv
from sme_negotiator_env.models import NegotiationAction


async def main():
    async with SMENegotiatorEnv(base_url="http://127.0.0.1:7860") as env:
        result = await env.reset(seed=42, task_name="payment-terms-easy")
        obs = result.observation

        action = NegotiationAction(
            action_type="propose",
            price=float(obs.buyer_price),
            payment_days=max(0, int(obs.buyer_days) - 5),
            use_treds=False,
            reason="Counter-offer to improve payment terms",
        )

        result = await env.step(action)
        print(result.reward, result.done)


asyncio.run(main())
```

If you want to call the OpenEnv API without installing the typed package:

```python
import asyncio
from openenv.core import GenericEnvClient


async def main():
    async with GenericEnvClient(base_url="http://127.0.0.1:7860") as env:
        await env.reset()
        await env.step(
            {
                "action_type": "propose",
                "price": 100.0,
                "payment_days": 80,
            }
        )


asyncio.run(main())
```

## Task Ladder

| Task | Opening state | What the agent must do | Full-credit signal |
|------|---------------|------------------------|--------------------|
| `payment-terms-easy` | Buyer opens at `100 INR / 90 days` | Compress terms toward the SME liquidity threshold | Reach a deal with agreed days `<= 60` |
| `payment-terms-medium` | Buyer opens at `100 INR / 60 days` | Tighten terms and use a late-payment penalty clause | Reach a deal with agreed days `<= 45`, with stronger partial credit if the clause is included |
| `payment-terms-hard` | Buyer opens at `96 INR / 100 days` in a hostile two-buyer setting | Negotiate dynamic discounting and manage financing tradeoffs | Improve NPV versus the status quo with `propose_dynamic_discounting=true` |

Important hard-mode notes:

- `use_treds=true` changes the simulation by lowering the buyer day floor.
- Hard-task terminal score is **not** based on TReDS alone.
- To earn hard-task credit, the agent must negotiate **dynamic discounting** with a sensible annual rate.

## How It Works

```text
LLM / heuristic / RL agent
          |
          v
SMENegotiatorEnv or GenericEnvClient
          |
          v
FastAPI + OpenEnv app  (port 7860)
          |
          v
SMENegotiatorEnvironment
          |
          +-- task_config.py
          +-- graders.py
          +-- typed state / observation models
```

The environment exposes:

- Partial step rewards for negotiation progress.
- Deterministic terminal rewards from task-specific graders.
- Rich observations with buyer terms, liquidity thresholds, working-capital gap, buyer power, and financing context.
- A state model that can be serialized for evaluation and replay.

## API And Action Surface

Main action fields:

- `action_type`: `propose`, `accept`, or `reject`
- `price`: proposed price in INR per unit
- `payment_days`: proposed settlement timeline
- `use_treds`: whether the SME invokes TReDS financing
- `reason`: optional text explanation
- `propose_late_payment_penalty_clause`: medium-task clause toggle
- `propose_dynamic_discounting`: hard-task financing toggle
- `dynamic_discount_annual_rate`: annualized early-payment discount rate

The default app is created in [`server/app.py`](server/app.py), backed by [`server/environment.py`](server/environment.py).

## Baseline Inference

[`inference.py`](inference.py) runs the environment with an OpenAI-compatible chat model. By default it targets the Hugging Face router, but you can also point it at a local server such as Ollama or LM Studio.

### Core environment variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | Hugging Face token for router-backed inference |
| `API_KEY` | Accepted as an alias for `HF_TOKEN` |
| `API_BASE_URL` | OpenAI-compatible LLM endpoint |
| `MODEL_NAME` | Chat-capable model id used by `chat/completions` |
| `OPENENV_BASE_URL` | Negotiation server URL, default `http://127.0.0.1:7860` |
| `OPENENV_IN_PROCESS` | Set to `1` to skip the server and run in-process |
| `TASK_FILTER` | Optional comma-separated subset such as `EASY,MEDIUM` |
| `INFERENCE_SKIP_LLM_AFTER_402` | Optional fallback guard after the first Hugging Face billing error |
| `INFERENCE_HARD_TWO_STEP` | Debug-only hard-mode accept shortcut; off by default |

Copy the sample env file first:

```bash
cp .env.example .env
# PowerShell: Copy-Item .env.example .env
```

For local LLM development, point `API_BASE_URL` at your local OpenAI-compatible server and leave `HF_TOKEN` empty if that server does not require a key.

## Logging And Evaluation Contract

The baseline runner emits parser-safe lines for evaluators:

- `[START]` once per episode with task and model metadata
- `[STEP]` after every action with serialized action, reward, done flag, and error field
- `[END]` once per episode with success flag, step count, and reward trace

This makes it easy to audit rollouts and compare agent behavior across seeds.

## Deployment

### Local server

```bash
uv run server
```

### Docker

```bash
docker build -f docker/Dockerfile -t openenv-sme-negotiator:latest .
docker run -p 7860:7860 openenv-sme-negotiator:latest
```

### Hugging Face Space

The repository is configured with Space frontmatter and Docker runtime metadata. The public Space for this environment is:

- https://huggingface.co/spaces/Omkarchaithanya/sme-negotiator

## Design Principles

- **Deterministic scoring:** task graders live in the environment and do not depend on LLM judgment.
- **Economic realism:** observations include working-capital stress, buyer leverage, supplier-pay timing, and financing cost.
- **Agent-facing clarity:** typed models and a typed client make the benchmark easy to plug into LLM, RL, or heuristic policies.
- **Auditability:** server behavior, reward logic, and task constants are easy to inspect in code.

## Repository Layout

```text
openenv-sme-negotiator/
|-- docker/
|   `-- Dockerfile
|-- server/
|   |-- app.py
|   |-- concurrency.py
|   |-- environment.py
|   `-- sme_environment.py
|-- sme_negotiator_env/
|   |-- client.py
|   |-- graders.py
|   |-- llm_action_parser.py
|   |-- models.py
|   |-- problem_context.py
|   `-- task_config.py
|-- tests/
|   |-- test_environment.py
|   `-- test_inference_policy.py
|-- inference.py
|-- openenv.yaml
|-- pyproject.toml
|-- SETUP.md
|-- EVALUATION.md
`-- TROUBLESHOOTING.md
```

## Validation And Tests

Run the test suite:

```bash
uv run pytest tests/ -v
```

Quick diagnostic:

```bash
make diagnostic
```

## References

- [Razorpay Fix My Itch](https://razorpay.com/m/fix-my-itch/)
- [OpenEnv docs](https://huggingface.co/docs/openenv)
- [MSME Samadhaan portal](https://samadhaan.msme.gov.in/MyMsme/MSEFC/MSEFC_Welcome.aspx)
- [Economic Survey 2025-26, Chapter 8](https://www.indiabudget.gov.in/economicsurvey/doc/eschapter/echap08.pdf)
- [RBI FAQ: TReDS](https://www.rbi.org.in/scripts/FAQView.aspx/FAQView.aspx/FAQView.aspx?Id=132)
- [MSMED delayed-payment context and buyer liability](https://samadhaan.msme.gov.in/MyMsme/MSEFC/MSEFC_Welcome.aspx)
- [TReDS ecosystem context](https://www.tredsindia.com/)

## Contributing

Contributions are welcome. High-value improvements include:

- new agent baselines
- reward-analysis tooling
- additional negotiation task variants
- replay and observability tooling
- benchmark reports and documentation improvements

For setup details, see [SETUP.md](SETUP.md). For evaluation notes, see [EVALUATION.md](EVALUATION.md).

## Citation

If you use this environment in research or benchmarking, cite it as:

```bibtex
@software{openenv_sme_negotiator,
  title = {OpenEnv SME Negotiator: B2B Payment-Term Negotiation Environment},
  year = {2026},
  url = {https://github.com/SkandaGanesha1/ENV}
}
```

## License

MIT. See [LICENSE](LICENSE).
