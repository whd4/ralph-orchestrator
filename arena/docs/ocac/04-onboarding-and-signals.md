# OCAC — Onboarding & Behavioral Signals

> The "Magic Moment" — where OCAC goes from generic AI to YOUR cognitive twin.

---

## Onboarding Flow (The Magic Moment)

### Step 1: Conversational Identity Mapping
Not a form. Not a quiz. A natural conversation that maps:

- **Personality**: How you think, communicate, make decisions
- **Goals**: What you're working toward (short-term and long-term)
- **Stress Profile**: What triggers you, how you cope, what helps
- **Motivation**: What drives you — achievement, autonomy, connection, mastery
- **Communication Style**: How you prefer to give and receive information

Duration: 15-30 minutes
Feeling: Like talking to a smart friend who's genuinely curious about you

### Step 2: Device & Platform Migration
User-controlled data ingestion from:

| Source | What OCAC Learns |
|--------|-----------------|
| Phone contacts | Relationship graph seed |
| Calendar | Work rhythm, commitments, energy patterns |
| Notes apps | Thought patterns, interests, project history |
| Email (opt-in) | Communication style, relationship dynamics |
| Social media (opt-in) | Public persona, interests, social patterns |
| Browser history (opt-in) | Interests, research patterns, time usage |
| Messages (opt-in) | Conversation style, relationship dynamics |

**Critical**: Every source is explicitly opt-in. User sees exactly what's being ingested. Can pause/stop at any time.

### Step 3: Memory State Initialization
From the conversation + ingested data, OCAC builds the initial:

- Preference vectors
- Decision pattern seeds
- Risk tolerance estimate
- Emotional baseline
- Daily rhythm model
- Communication style profile

This is the "v0" of the Personal Memory State — rough but functional. It refines continuously from every interaction.

---

## Behavioral Signals (Opt-In)

### Always Available (No Special Hardware)
| Signal | Source | What It Tells OCAC |
|--------|--------|-------------------|
| Typing patterns | Keyboard | Energy level, stress, engagement |
| Response latency | Messages | Cognitive load, interest, avoidance |
| Calendar density | Calendar app | Stress load, available bandwidth |
| App usage patterns | Device analytics | Focus, distraction, interests |
| Communication frequency | All channels | Social energy, relationship investment |

### Wearable Signals (Smartwatch/Fitness Tracker)
| Signal | What It Tells OCAC |
|--------|-------------------|
| Heart rate trends | Stress response, excitement, calm states |
| Heart rate variability | Resilience, recovery, autonomic balance |
| Sleep/wake patterns | Energy cycle, recovery quality, rhythm disruption |
| Activity levels | Physical energy, sedentary alerts, exercise correlation with mood |
| Step count patterns | Routine stability, disruption signals |

### Advanced Signals (Future)
| Signal | What It Tells OCAC |
|--------|-------------------|
| Voice tone analysis | Real-time emotional state during conversations |
| Eating patterns | Routine stability, stress-eating, health habits |
| Location patterns | Routine, novelty-seeking, social behavior |
| Spending patterns | Financial stress, impulse behavior, priorities |

### Signal Processing Rules
1. All signals are opt-in (granular — per signal, not all-or-nothing)
2. Raw data is processed locally, only insights are stored
3. User can see exactly what OCAC infers from each signal
4. Signals are NEVER used for medical/health claims
5. User can disable any signal at any time (existing inferences remain unless explicitly deleted)
6. Signal correlation is transparent: "I noticed your heart rate was elevated during that meeting. Want to talk about it?"
