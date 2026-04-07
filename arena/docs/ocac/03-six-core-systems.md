# OCAC — Six Core Systems

> These are the six systems that make OCAC a cognitive twin. Each is independently valuable but exponentially powerful when combined.

---

## 1. Behavioral Baseline Test (BBT)

**Purpose**: Non-clinical fingerprinting of the user's behavioral identity.

### Dimensions Measured
| Dimension | What It Captures |
|-----------|-----------------|
| Decision Style | Analytical vs. intuitive, fast vs. deliberate, risk-seeking vs. risk-averse |
| Communication Tone | Formal vs. casual, direct vs. indirect, verbose vs. terse |
| Stress Patterns | Triggers, coping mechanisms, escalation timeline, recovery patterns |
| Motivation Drivers | Achievement, autonomy, mastery, purpose, social connection |
| Social Energy | Introvert/extrovert spectrum, recharge patterns, social battery indicators |
| Conflict Style | Avoidant, competitive, collaborative, accommodating, compromising |
| Work Rhythm | Peak hours, focus duration, task-switching tolerance, deadline behavior |
| Emotional Cadence | Baseline mood, variability, triggers for highs/lows, recovery speed |

### How It Works
- Conversational assessment (not a quiz — feels like a natural conversation)
- Continuously refined over time (never "done")
- User can view and correct any assessment
- No medical/psychological claims — behavioral patterns only

### Off-Baseline Detection
When current behavior deviates significantly from baseline:
- Flag the deviation (don't diagnose)
- Ask if user wants to explore it
- Adjust recommendations to account for current state
- Log for pattern analysis (with consent)

---

## 2. Behavioral Memory Architecture (BMA)

**Purpose**: Six-layer memory system that gives OCAC persistent, evolving knowledge of the user.

### The 6 Layers

| Layer | What It Stores | Retention |
|-------|---------------|-----------|
| **Identity** | Core traits, values, goals, non-negotiables | Permanent (user-editable) |
| **Behavior** | Decision patterns, habits, preferences, routines | Long-term, slowly evolving |
| **Interaction** | Conversation history, tool usage, response patterns | Medium-term, rolling window |
| **Relationship** | People graph, interaction quality, reciprocity scores | Long-term, event-updated |
| **Outcome** | Decision results, feedback, success/failure patterns | Permanent (append-only) |
| **Context** | Current state — time, location, mood, energy, calendar | Ephemeral, real-time |

### Properties
- Local-first, encrypted
- Each layer has independent read/write permissions
- User can freeze any layer (stop updates)
- Export/import supported (portable identity)
- Conflict resolution: newer data wins, but old data is archived (not deleted)

---

## 3. Conversation Insight Engine (CIE)

**Purpose**: Real-time and post-conversation analysis of communication patterns.

### Real-Time Analysis
- **Transcription**: Speech-to-text with speaker diarization
- **Tone Detection**: Sentiment, emotion, energy level per utterance
- **Interruption Patterns**: Who interrupts whom, frequency, impact on conversation
- **Dominance Mapping**: Speaking time distribution, topic control, question/answer ratio
- **Reciprocity Score**: Balance of give-and-take in the conversation

### Post-Conversation Coaching
After each analyzed conversation:
1. Summary of key dynamics
2. Your communication patterns in this conversation
3. How they compare to your baseline
4. Specific, actionable suggestions
5. Relationship impact assessment

### Privacy Controls
- User chooses which conversations to analyze
- Other participants must consent (for live analysis)
- Raw transcripts can be deleted while keeping aggregated insights
- Coaching suggestions are private to the user

---

## 4. Relationship Pattern Analyzer (RPA)

**Purpose**: Map and monitor the user's relationship network for health and patterns.

### What It Tracks (Per Relationship)
| Signal | What It Means |
|--------|--------------|
| Frequency | How often you interact — increasing, decreasing, stable |
| Tone | Average emotional tone of interactions — positive, negative, neutral |
| Emotional Impact | How you feel after interactions — energized, drained, neutral |
| Reciprocity | Balance of initiation, response time, effort |
| Stress Spikes | Interactions that correlate with elevated stress indicators |
| Avoidance Patterns | Declining invitations, delayed responses, topic avoidance |
| Positive Loops | Interactions that consistently produce good outcomes |
| Negative Loops | Patterns that repeat without resolution |

### Outputs
- **Relationship health dashboard** (visual overview)
- **Early warning system** (drift detection before problems become crises)
- **Suggestions**: "You haven't talked to [person] in 3 weeks. You usually feel better after catching up."
- **Boundary alerts**: "This relationship shows a pattern of one-sided effort."

### Ethics
- Never tells user what to do about relationships — only surfaces patterns
- Never shares relationship data with the other person
- User controls which relationships are tracked

---

## 5. Negotiation & Communication Coach (NCC)

**Purpose**: Help the user communicate more effectively in high-stakes situations.

### Capabilities
| Feature | Description |
|---------|-------------|
| **Message Drafting** | Generate message options in user's voice with adjustable tone |
| **Tone Rewriting** | Rewrite existing messages to be more assertive, diplomatic, or neutral |
| **Assertiveness Coaching** | Practice saying no, setting boundaries, making requests |
| **Conflict De-escalation** | Reframe heated messages, suggest cooling-off strategies |
| **Boundary Setting** | Help articulate and enforce personal/professional boundaries |
| **Negotiation Prep** | Simulate counterarguments, identify leverage points, practice responses |

### How It Works
1. User provides context (who, what, stakes, desired outcome)
2. NCC generates options ranked by communication style match
3. User picks or modifies
4. NCC tracks outcomes to improve future suggestions

### Voice Matching
- Uses TPE personality dimensions to match user's natural voice
- Adapts formality level to the relationship
- Can switch modes: "write this like I'd say it" vs. "write this professionally"

---

## 6. Twin Personality Engine (TPE)

**Purpose**: Make the twin feel like a real person — not a tool, not a generic assistant.

### 8 Personality Dimensions
| Dimension | Spectrum |
|-----------|----------|
| Tone | Formal ←→ Casual |
| Humor | Dry/Witty ←→ Silly/Playful |
| Directness | Blunt ←→ Diplomatic |
| Warmth | Cool/Professional ←→ Warm/Affectionate |
| Energy | Calm/Measured ←→ Enthusiastic/High-energy |
| Playfulness | Serious ←→ Mischievous |
| Seriousness | Light ←→ Grave |
| Challenge Level | Supportive ←→ Provocative |

### 8 Communication Modes
| Mode | When to Use |
|------|------------|
| **Supportive** | User needs encouragement, validation, emotional support |
| **Strategic** | User needs analysis, planning, decision-making help |
| **Playful** | Casual conversation, stress relief, creative brainstorming |
| **Tough Love** | User is making excuses, needs honest push |
| **Sarcastic** | User enjoys banter, witty back-and-forth (opt-in only) |
| **Calm** | User is stressed, needs grounding, de-escalation |
| **High-Energy** | User needs motivation, hype, momentum |
| **No-Nonsense** | User wants facts, no fluff, pure efficiency |

### Adaptation Rules
- Default personality is set during onboarding (BBT results)
- Mode switches automatically based on context (can be overridden)
- User can say "be more direct" or "lighten up" at any time
- The twin REMEMBERS personality preferences (stored in BMA Identity layer)
- Can push back, joke, challenge, even cuss (per user preference)
- Never breaks character unless user explicitly requests it
