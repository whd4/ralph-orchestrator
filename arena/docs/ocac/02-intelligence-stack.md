# OCAC Intelligence Stack

> The core reasoning engine that makes OCAC a cognitive twin, not just a chatbot.

---

## Architecture Overview

```
User Input / Behavioral Signal
        │
        ▼
┌─────────────────────────┐
│  Personal Memory State   │  ← Encrypted, local-first, per-user identity
│  (behavioral baseline)   │
└────────────┬────────────┘
             │
     ┌───────┴───────┐
     ▼               ▼
┌─────────┐   ┌──────────────┐
│  Deep    │   │  Monte Carlo │
│Confidence│   │  Prediction  │
│  Engine  │   │    Engine    │
└────┬─────┘   └──────┬──────┘
     │                 │
     └────────┬────────┘
              ▼
     ┌────────────────┐
     │ Path Generator  │  ← Recommended action + tradeoffs
     └────────────────┘
```

---

## 1. Deep Confidence Engine (DC)

**Purpose**: Rank possible actions by confidence level, cut low-confidence branches early.

### How It Works
1. Takes user context + memory state as input
2. Evaluates each possible action against behavioral patterns
3. Assigns confidence scores (0.0 - 1.0) based on:
   - Historical decision patterns
   - User's stated preferences
   - Outcome history for similar decisions
   - Current emotional/stress state
4. Produces a **weighted shortlist** of high-confidence actions
5. Low-confidence branches are pruned (not deleted — stored for learning)

### Key Properties
- Confidence thresholds are per-user (adapts over time)
- Explains WHY confidence is high/low (transparency)
- Never makes decisions autonomously without user-set thresholds

---

## 2. Monte Carlo Prediction Engine (MC)

**Purpose**: Run thousands of micro-simulations to predict outcome distributions.

### How It Works
1. Takes the DC shortlist as input
2. For each high-confidence action, simulates 1,000+ scenarios
3. Variables include:
   - User's typical response patterns
   - External factors (time pressure, relationship dynamics, market conditions)
   - Historical outcomes of similar decisions
   - Randomized perturbations (what-if scenarios)
4. Produces **probability distributions** for each outcome
5. Identifies tail risks (low-probability, high-impact scenarios)

### Output Format
- Expected outcome (median)
- Best case (95th percentile)
- Worst case (5th percentile)
- Tail risks flagged separately
- Confidence interval for each prediction

---

## 3. Personal Memory State (PMS)

**Purpose**: Encrypted, local-first behavioral identity that makes OCAC uniquely yours.

### What It Stores
- Decision history (what you chose, when, why)
- Preference vectors (communication style, risk tolerance, work rhythm)
- Emotional baseline (stress patterns, energy cycles, mood indicators)
- Relationship graph (who matters, interaction quality, reciprocity)
- Outcome feedback (what worked, what didn't, user ratings)

### Properties
- **Local-first**: Never leaves the user's device without explicit consent
- **Encrypted at rest**: AES-256 or equivalent
- **Continuous learning**: Updates after every interaction
- **User-controlled**: Can view, edit, export, or delete any memory
- **No cross-user sharing**: Each PMS is an isolated universe

---

## 4. Path Generator

**Purpose**: Combines DC + MC outputs to produce a recommended path with transparent tradeoffs.

### How It Works
1. Takes DC confidence scores + MC simulation results
2. Weights actions by: confidence * expected_outcome * user_preference_alignment
3. Produces a ranked list with:
   - **Recommended path** (highest weighted score)
   - **Alternative paths** (other viable options)
   - **Tradeoff analysis** (what you gain/lose with each choice)
   - **Risk assessment** (tail risks from MC simulations)
4. Presents in user's preferred communication style (from TPE)

### Decision Modes
- **Advisory**: Shows recommendations, user decides (default)
- **Semi-autonomous**: Acts on high-confidence decisions, asks for others
- **Autonomous**: Acts on all decisions above user-set threshold
