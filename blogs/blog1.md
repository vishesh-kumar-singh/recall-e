## Blog Metadata
- **Title:** Teaching Machines to Learn from Experience: MDPs and Q-Learning
- **Authors:** Tejas, Saurav
- **Week:** Week 1
- **Date:** May 23, 2026
- **Short Summary:** An intuitive walkthrough of Reinforcement Learning, Markov Decision Processes, the Bellman equation, and Q-Learning — with code along the way.

---

### Introduction

Imagine teaching a dog to sit. You don't hand it a manual. You don't show it a thousand labeled photos of "sitting dog." You just say "sit", wait, and give it a treat when it gets it right. Over many tries, the dog figures out what earns the treat and starts doing it reliably.

That, in spirit, is Reinforcement Learning (RL).

This week, we're diving into the foundations of RL that power a huge chunk of what the Recall-E project is trying to do — from Markov Decision Processes to Q-Learning. No heavy math, just the intuition. Let's go.

---

### What Even Is Reinforcement Learning?

Before we get into RL, let's quickly look at the ML landscape so we know what we're contrasting with.

**Supervised Learning** is like studying with an answer key. You give the model input-output pairs — "this image is a cat, this one is a dog" — and it learns to map inputs to correct outputs. The labels are handed to it.

**Unsupervised Learning** is like exploring a new city with no map. You give the model raw data with no labels, and it tries to find hidden structure — clusters, patterns, anomalies — on its own.

**Reinforcement Learning** is neither. There's no answer key. There's no pre-labeled data. Instead, there's an **agent** (the learner), an **environment** (the world it acts in), and a **reward signal** (a score that tells it how well it's doing *after the fact*).

The agent tries things, sees what happens, gets a reward (or a penalty), and gradually learns which actions lead to better outcomes. It's learning by doing — trial, error, and feedback.

**The Key Difference: In supervised learning, you're told the right answer. In RL, you only know *how good* your answer was, and sometimes only much later.**

A few classic examples where RL shines:
- An agent learning to play chess (the reward only comes at the end — win or lose)
- A robot learning to walk (small rewards for staying upright, penalties for falling)

---

### Markov Decision Processes (MDPs)

Okay, so RL is about an agent learning in an environment. But to actually *build* and *reason about* these systems, we need a formal framework. That framework is the **Markov Decision Process**, or MDP.

Think of an MDP as the rulebook of the game. It has four main ingredients:

#### States (S) — "Where am I?"

A **state** is a snapshot of the world at a given moment. In a chess game, the state is the current position of all pieces on the board. In a video game, it might be your character's position, health, and what enemies are nearby.

The key assumption in an MDP is the **Markov Property**: *the future depends only on the present state, not on how you got there.* In other words, yesterday's moves don't matter — only where things stand right now. This is a simplification, but a powerful one.

#### Actions (A) — "What can I do?"

From any state, the agent can choose an **action**. In chess, actions are the legal moves. In a grid world, maybe it's {up, down, left, right}.

#### Transitions (T) — "What happens next?"

When you take an action, something happens — you move to a new state. Simple as that. Sometimes the world is predictable: press right, move right. But sometimes it's not — like a robot walking on a slippery floor. You try to go right, but maybe you slip and end up somewhere else. Transitions capture exactly this: what state do you land in after taking an action?

#### Rewards (R) — "Was that good?"

After every transition, the environment gives the agent a **reward** — a number that can be positive (great move!) or negative (walked into a wall). The agent's job is to learn a **policy** (a strategy: "given state X, take action Y") that *maximizes the total reward it accumulates over time*.

Here's a tiny code sketch of an MDP-like setup using OpenAI Gym's FrozenLake:

```python
import gymnasium as gym

# A simple grid-world environment
# The agent must reach the goal without falling into holes
env = gym.make("FrozenLake-v1", is_slippery=True)

state, _ = env.reset()
print(f"Starting state: {state}")

# Take a random action (0=Left, 1=Down, 2=Right, 3=Up)
action = env.action_space.sample()
next_state, reward, terminated, truncated, info = env.step(action)

print(f"Action: {action}, Next State: {next_state}, Reward: {reward}")
```

---


### Exploration vs. Exploitation: The Great Balancing Act

Imagine it's movie night.

- **Exploitation** is rewatching your favorite show. You know exactly what you're getting, and you're guaranteed a good time. You're playing it safe with what you already know.
- **Exploration** is picking a random, unknown movie. It might be awful and ruin your night — but it could also become your new all-time favorite!

An AI faces this exact same choice every time it makes a move.

If it only plays safe (exploits), it might get stuck with a low score forever because it never looked for a better path. If it only tries random things (explores), it never actually uses what it learned to win.

The secret is **balance**. A good AI starts by exploring wildly to learn how its world works. Then, over time, it slowly shifts to exploiting that knowledge to get the highest score possible.

This is captured by the **ε-greedy strategy** (epsilon-greedy): with probability ε, pick a random action (explore); otherwise, pick the best known action (exploit). As training progresses, ε is gradually decayed — you'll see this in the Q-Learning code below.

**The Main Point: Too much exploitation = you get stuck in a local optimum. Too much exploration = you never cash in on what you've learned. Great RL agents learn when to be bold and when to be disciplined.**

---

### The Bellman Equation: Thinking About the Future

A smart agent doesn't just care about the reward right in front of it. It thinks ahead. A small loss now might be worth a big win later.

That's the core idea behind the **Bellman Equation**. In simple words:

**The value of being somewhere = the reward you get now + the best reward you can get from here onwards.**

As a formula:
```
Value(state) = Immediate Reward + γ × Value(best next state)
```

That `γ` (gamma) is called the **discount factor**. It's a number between 0 and 1 that controls how much the agent cares about future rewards. A gamma close to 1 means the agent is patient — future rewards matter a lot. A gamma close to 0 means the agent is greedy — it only cares about right now.

Think of it this way: ₹100 today feels more valuable than ₹100 a year from now. Gamma captures exactly that.

Now in Q-Learning, instead of just asking "how good is this state?", we ask "how good is taking this specific action in this state?" That's what a **Q-value** is. Written out:

$$
Q(s, a) = r + \gamma \max_{a'} Q(s', a')
$$

In plain words: the score for taking action `a` in state `s` is the reward you got, plus the best score possible from the next state.

#### Why Do We Need a Learning Rate?

Early in training, the agent has no idea what the future looks like. The Q-table starts at zero, which means those future Q-values it's relying on are basically just guesses. If we fully trusted the Bellman equation from the start, we'd be computing answers from wrong inputs — and those errors would keep piling up and making things worse.

So instead of jumping straight to the answer, we take small steps. That's what the **learning rate α** is for. Each time the agent learns something new, it doesn't throw out its old belief completely — it just nudges it a little in the right direction:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

Breaking it down:
- `r` — the reward the agent just got
- `max_a' Q(s', a')` — the best Q-value from the next state (the agent's current guess about the future)
- `α` (alpha) — the learning rate. High alpha means "update a lot from this." Low alpha means "only shift a little."
- The part in brackets — how far off the old estimate was

It's like adjusting your opinion slowly. You hear one new piece of information and you don't completely change your mind — you just update a little. Do that thousands of times, and eventually you get pretty close to the truth.

---

### Q-Learning: Putting It All Together

Q-Learning is an algorithm that uses the Bellman equation repeatedly to learn the optimal policy. Here's the big picture:

1. Build a **Q-table** — a big grid where rows are states and columns are actions. Initialize everything to zero.
2. For each step: observe state → pick action (ε-greedy) → get reward → observe new state → update Q-table using the Bellman equation.
3. Repeat millions of times until the Q-values stabilize.

Once trained, the policy is simple: in any state, just pick the action with the highest Q-value.

Here's a full Q-Learning implementation on FrozenLake:

```python
import gymnasium as gym
import numpy as np

# Environment setup
env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n   # 16 states in a 4x4 grid
n_actions = env.action_space.n       # 4 actions: Left, Down, Right, Up

# Hyperparameters
alpha = 0.8      # Learning rate
gamma = 0.95     # Discount factor
epsilon = 1.0    # Starting exploration rate
epsilon_decay = 0.001
min_epsilon = 0.01
n_episodes = 10000

# Initialize Q-table to zeros
q_table = np.zeros((n_states, n_actions))

# Training loop
for episode in range(n_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        # ε-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        # Take action, observe result
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Bellman update
        best_next_q = np.max(q_table[next_state])
        td_error = reward + gamma * best_next_q - q_table[state, action]
        q_table[state, action] += alpha * td_error

        state = next_state

    # Decay epsilon (explore less as we learn more)
    epsilon = max(min_epsilon, epsilon - epsilon_decay)

print("Training complete!")
print("Q-table:\n", q_table)

# Quick evaluation
wins = 0
for _ in range(100):
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state])  # Always exploit during eval
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if reward == 1:
            wins += 1

print(f"Win rate after training: {wins}%")
```

After enough training, the agent learns a solid policy for navigating the grid world — purely through trial and error, no labeled data, no explicit programming of the rules.

---

### Drawbacks of Q-Learning

Q-Learning is a great starting point, but it has some real pain points. Here's where it starts to struggle:

**1. It can't handle big, complex worlds**

The Q-table stores one entry for every possible state. For a tiny grid world, that's totally fine. But imagine trying to do this for a video game — the number of possible screen states is so massive, no table could ever hold it all. It's like trying to write down every possible position every grain of sand on a beach could be in. Just not practical.

**2. It only works when choices are clear-cut**

Q-Learning is great when actions are simple and countable — go left, go right, jump. But what if the action is something like "how hard should I push this door?" That's not a list of options anymore, it's a sliding scale. Q-Learning has no way to handle that kind of decision.

**3. It needs a LOT of practice**

Q-Learning learns by doing — thousands, sometimes millions of tries. In a video game simulation, that's fine, you just let it run overnight. But imagine a robot in the real world that needs to fall over a million times before it learns to walk. Not great. Every mistake costs time, money, or in some cases, could even cause harm.

**4. It assumes you can see everything**

Q-Learning works best when the agent has the full picture of what's going on. But real life is messier. A robot can't see around corners. A doctor can't know everything about a patient from a few test results. When the agent is working with incomplete information, Q-Learning starts making poor decisions.

**5. It does exactly what you tell it — even if that's not what you meant**

You'd think giving an AI a reward signal would be straightforward. But agents are sneaky — they'll find any loophole to maximize that score, even if it means doing something completely unintended. There's a famous example of an RL agent in a boat racing game that figured out it could score more points by spinning in circles collecting bonuses than by actually finishing the race. Technically correct. Completely useless.

**6. It tends to be overconfident**

Q-Learning has a habit of thinking things are better than they actually are. When it picks the "best" action during training, it slightly inflates how good that action looks in hindsight. Over time, these small overestimates pile up, and the agent ends up chasing a distorted picture of reality. A smarter version called Double Q-Learning was built specifically to fix this blind spot.

---

### Conclusion

Let's recap the journey:

- **RL** is a learning paradigm where an agent learns by interacting with an environment — no labels needed, just rewards.
- **MDPs** give us a clean formalism: states, actions, transitions, and rewards. They're the rulebook every RL algorithm is playing by.
- The **exploration-exploitation tradeoff** is the eternal question: do you stick with what works, or gamble on something better?
- The **Bellman equation** is the recursive insight that value = immediate reward + discounted future value. It's the engine behind almost all RL algorithms.
- **Q-Learning** puts it all together into a practical algorithm that fills a Q-table through repeated experience.
- But Q-Learning **breaks down** at scale — large state spaces, continuous actions, partial observability, and sample inefficiency are all open problems that motivated the modern deep RL revolution.
