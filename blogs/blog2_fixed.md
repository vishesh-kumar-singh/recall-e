# Week 2 Blog
**By:** Prakhar Agarwal, Priyanka Chadokar, Harpal Singh, Hemendra Gurjar

## Introduction
In this blog, we will unpack the core components of modern value-based Reinforcement Learning (RL), tracing how a series of incremental breakthroughs led to state-of-the-art performance. We will cover:

1. **Deep Q-Network (DQN):** The foundational algorithm that bridged the gap between classic Q-learning and deep neural networks.
2. **Double DQN (DDQN):** The mathematical correction for standard DQN's inherent tendency to overestimate action values.
3. **Prioritized Experience Replay (PER):** An optimization for DQN and DDQN that samples transitions based on their learning value, making training significantly more sample-efficient.

We will also briefly explore the mechanics of **Dueling Networks**, **Noisy Nets**, **Distributional RL**, and **N-step Learning**, before concluding with a bird's-eye view of **Rainbow DQN**—the ultimate architecture that unites all of these pieces into a single, powerhouse agent.

## Deep Q-Network (DQN)
In the case of a continuous state, it is computationally extremely heavy to even store a Q-value for every possible state and action, let alone try to train an agent on this Q-Table.

### How do we tackle this problem?
The answer lies in a very fundamental and famous concept of Machine Learning: **Neural Networks**.

Using a neural network, we can input the state as a continuous choice of numbers and receive a particular value for every possible action $a \in A$. This value corresponds to the action-value function. The neural network is said to have weights $\theta$, and this neural network is called a **Q-Network**.

Thus, we redefine learning as training the Q-Network (modifying the weights $\theta$) in a similar—but not identical—manner to neural networks in supervised learning.

### Defining a Loss Function
A Q-network can be trained by minimizing a sequence of loss functions $L_i(\theta_i)$ that changes at each iteration $i$:

$$L_i(\theta_i) = \mathbb{E}_{(s,a,r,s') \sim U(D)} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta_i^-) - Q(s, a; \theta_i) \right)^2 \right]$$

> **Important Note:** Note that the targets depend on the network weights; this is in contrast with the targets used for supervised learning, which are fixed before learning begins. We also use the **Exploration vs. Exploitation** logic while choosing the action when in state $s$.

Note that this algorithm is **model-free**: it solves the reinforcement learning task directly using samples from the emulator $E$, without explicitly constructing an estimate of $E$.

### Experience Replay
We utilize a technique known as **experience replay** where we store the agent’s experiences at each time-step, $e_t = (s_t, a_t, r_t, s_{t+1})$, in a dataset $\mathcal{D} = \{e_1, \dots, e_N\}$, pooled over many episodes into a replay memory.

Before performing experience replay, the agent selects and executes an action according to an $\varepsilon$-greedy policy. Since using histories of arbitrary length as inputs to a neural network can be difficult, our Q-function instead works on a fixed-length representation of histories produced by a function $\phi$.

During the inner loop of the algorithm, we apply Q-learning updates, or minibatch updates, to samples of experience, $e \sim \mathcal{D}$, drawn at random from the pool of stored samples.

### Preprocessing
To convert our observation into input for our Q-network, we need to observe our state and make appropriate changes to suit our requirement of the model architecture. We will not get into the details of how this is done as it varies from environment to environment.

### Algorithm 1: Deep Q-learning with Experience Replay

---

1. **Initialize** replay memory $\mathcal{D}$ to capacity $N$
2. **Initialize** action-value function $Q$ with random weights $\theta$
3. **for** $\text{episode} = 1, M$ **do**
4. &nbsp;&nbsp;&nbsp;&nbsp;Initialize sequence $s_1 = \{x_1\}$ and preprocessed sequence $\phi_1 = \phi(s_1)$
5. &nbsp;&nbsp;&nbsp;&nbsp;**for** $t = 1, T$ **do**
6. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;With probability $\varepsilon$ select a random action $a_t$
7. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Otherwise select $a_t = \arg\max_a Q(\phi(s_t), a; \theta)$
8. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Execute action $a_t$ in emulator and observe reward $r_t$ and image $x_{t+1}$
9. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Set $s_{t+1} = s_t, a_t, x_{t+1}$ and preprocess $\phi_{t+1} = \phi(s_{t+1})$
10. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Store transition $(\phi_t, a_t, r_t, \phi_{t+1})$ in $\mathcal{D}$
11. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample random minibatch of transitions $(\phi_j, a_j, r_j, \phi_{j+1})$ from $\mathcal{D}$
12. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Set $y_j = \begin{cases} r_j & \text{for terminal } \phi_{j+1} \\ r_j + \gamma \max_{a'} Q(\phi_{j+1}, a'; \theta^-) & \text{for non-terminal } \phi_{j+1} \end{cases}$
13. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Perform a gradient descent step on $\left(y_j - Q(\phi_j, a_j; \theta)\right)^2$ with respect to the network parameters $\theta$
14. &nbsp;&nbsp;&nbsp;&nbsp;**end for**
15. **end for**

---

## The Code

### 1) The Q-Network Architecture
> **Note:** Do not worry too much about this right now as it changes from situation to situation; this is provided just for illustration purposes.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # A simple network structure for vector states
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        # Feed forward pass to get Q-values for all actions
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
```

### 2) The Randomized Experience Buffer
> **Note:** This is the formal code implementation for the Experience Replay Buffer algorithm described earlier.

```python
import random
from collections import deque

class ReplayMemory:
    def __init__(self, capacity):
        # Initialize replay memory D to capacity N
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Store transition (phi_t, a_t, r_t, phi_t+1, done) in D
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Sample random minibatch of transitions from D
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
```

### 3) The DQN Algorithm and Training Loop
> **Note:** This is the heart of the DQN algorithm and should be well understood.

```python
import numpy as np
import torch.optim as optim
import random
import torch
import torch.nn.functional as F

# --- Hyperparameters & Initialization ---
state_dim = 4      # Example state dimension
action_dim = 2     # Example action dimension
N = 10000          # Replay memory capacity
M = 500            # Number of episodes
T = 200            # Max steps per episode
batch_size = 32
gamma = 0.99       # Discount factor
epsilon = 0.1      # Exploration rate (fixed for simplicity)

# Initialize replay memory D to capacity N
D = ReplayMemory(N)

# Initialize action-value function Q with random weights
Q = QNetwork(state_dim, action_dim)
optimizer = optim.Adam(Q.parameters(), lr=1e-3)

# Dummy environment function placeholders
def preprocess(state): return torch.FloatTensor(state)
def select_random_action(): return random.randint(0, action_dim - 1)
def env_step(action): return np.random.rand(state_dim), random.random(), random.choice([True, False])
def env_reset(): return np.random.rand(state_dim)

# --- Main Training Loop ---
for episode in range(1, M + 1):
    raw_state = env_reset()
    phi = preprocess(raw_state)
    
    for t in range(1, T + 1):
        # Action selection (Epsilon-greedy)
        if random.random() < epsilon:
            action = select_random_action()
        else:
            with torch.no_grad():
                q_values = Q(phi)
                action = torch.argmax(q_values).item()

        # Execute action in emulator and observe reward and next state
        next_raw_state, reward, done = env_step(action)
        phi_next = preprocess(next_raw_state)

        # Store transition in D
        D.push(phi, action, reward, phi_next, done)

        # Move to next state
        phi = phi_next

        # Start training only if we have enough samples
        if len(D) < batch_size:
            if done: break
            continue

        # Sample random minibatch of transitions from D
        minibatch = D.sample(batch_size)

        states = torch.stack([b[0] for b in minibatch])
        actions = torch.LongTensor([b[1] for b in minibatch]).unsqueeze(1)
        rewards = torch.FloatTensor([b[2] for b in minibatch])
        next_states = torch.stack([b[3] for b in minibatch])
        dones = torch.FloatTensor([b[4] for b in minibatch])

        # Compute Q(phi_j, a_j; theta)
        current_q_values = Q(states).gather(1, actions).squeeze(1)

        # Compute targets y_j
        with torch.no_grad():
            max_next_q_values = Q(next_states).max(1)[0]
            targets = rewards + (1 - dones) * gamma * max_next_q_values

        # Perform a gradient descent step
        loss = F.mse_loss(current_q_values, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if done:
            break
```

## The Fatal Flaws: Why We Need Double DQN (DDQN)
There are two main reasons:

### 1. Moving Targets (Chasing Your Own Tail)
Remember the important note: we are using the exact same network weights to calculate both our prediction and our target. Because the target shifts every time the weights update, it is like a dog chasing its own tail—a dynamic that leads to massive training instability and divergence in non-linear function approximators.

### 2. Overestimation Bias (The "Hard Max" Problem)
Because our neural network is just estimating values, those estimates are noisy. Some estimates will be slightly too low, and some will be slightly too high. By taking the hard max ($\max_a$) over all possible actions, standard DQN systematically favors the positively biased estimates. It blindly assumes the highest estimated value is the true maximum value.

Because DQN bootstraps using its own overestimations to update previous states, this positive bias snowballs out of control. The network starts hallucinating that certain bad actions are actually fantastic, leading to catastrophic drops in performance. Standard DQN simply could not decouple **selecting** the best action from **evaluating** the worth of that action.

## Double Deep Q-Network (DDQN)

### The Original DDQN
What we usually study and are going to discuss more about is an easy-to-implement version of DDQN, but the general DDQN is actually given by (van Hasselt, 2010).

In the original Double Q-learning algorithm, two value functions are learned by assigning each experience randomly to update one of the two value functions, such that there are two sets of weights, $\theta$ and $\theta'$. For each update, one set of weights is used to determine the greedy policy and the other to determine its value. The Double Q-learning estimate can then be written as:

$$Y_t^{DoubleQ} \equiv R_{t+1} + \gamma Q(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta_t); \theta_t')$$

Notice that the selection of the action, in the $\arg\max$, is still due to the online weights $\theta_t$. This means that, as in Q-learning, we are still estimating the value of the greedy policy according to the current values, as defined by $\theta_t$.

However, we use the second set of weights $\theta'_t$ to fairly evaluate the value of this policy. This second set of weights can be updated symmetrically by switching the roles of $\theta$ and $\theta'$.

### The DDQN We Will Use
Instead of separately maintaining two completely independent weight vectors, we will use the online network weights $\theta$ and a target network with weights $\theta^-$. This is how we will proceed to decouple action selection from action evaluation.

#### 1. Action Selection
We ask our current, active **Online Network** ($\theta$) to pick the action it thinks is best for the next state:

$$a^* = \arg\max_a Q(S_{t+1}, a; \theta_t)$$

#### 2. Action Evaluation
Instead of blindly trusting the online network's valuation of $a^*$, we pass this chosen action to the **Target Network** ($\theta^-$) to get its objective value:

$$Q(S_{t+1}, a^*; \theta_t^-)$$

#### 3. The Double DQN Target Formula 
When we merge these two steps into our loss function, our new TD target becomes:

$$Y_t^{DoubleDQN} \equiv R_{t+1} + \gamma Q(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta_t); \theta_t^-)$$

#### 4. Loss Minimization & Target Updates
Using this target, we find the Mean Squared Error (MSE) just as we did before, and **only update the online network** $\theta$.

It is only after a fixed number of iterations that we synchronize the networks by copying the weights of the online network ($\theta$) over to the target network ($\theta^-$).

Because the weights of $\theta$ and $\theta^-$ are distinct, a random positive noise spike for an action in the online network is highly unlikely to match a positive noise spike for that same action in the target network. The overestimation bias effectively vanishes. As we are not using the same weights to determine both the greedy policy and its value, the agent is also no longer "chasing its own tail".

> **Note:** The code for DDQN requires only minor additions and changes to the standard DQN implementation (specifically tracking and updating the target network), so it is not discussed separately here.

## Prioritized Experience Replay (PER)

We have already seen experience replay. Experience replay lets online reinforcement learning agents remember and reuse experiences from the past. In prior work, experience transitions were uniformly sampled from a replay memory. However, this approach simply replays transitions at the same frequency that they were originally experienced, regardless of their significance.

The real addition of value comes when we develop a framework for prioritizing experience, so as to replay important transitions more frequently, and therefore learn more efficiently.

Before considering Prioritized Experience Replay, let’s consider what happens if we don’t even consider experience replay (i.e., discarding incoming data immediately).

### 1. Strongly Correlated Updates
Discarding data immediately forces the agent to learn only from its most recent actions. This results in strongly correlated updates that break the **i.i.d. assumption** (Independent and Identically Distributed) foundational to many popular stochastic gradient-based algorithms.

### 2. Rapid Forgetting of Rare Experiences
Without a storage mechanism, the agent suffers from the rapid forgetting of possibly rare experiences that would be incredibly useful later on. If a critical or rare reward state is only visited once every few thousand steps, the network will quickly overwrite what it learned from that encounter during subsequent, mundane steps.

### Benefits of Standard Experience Replay 
Even without considering prioritization, introducing a standard, uniform experience replay buffer provides massive architectural benefits:

In general, experience replay can reduce the amount of experience required to learn, and replace it with more computation and more memory—which are often cheaper resources than the RL agent’s interactions with its physical or simulated environment.

We discuss how prioritizing some transitions which are replayed can make experience replay more efficient and effective than if all transitions are replayed uniformly. The key idea is that an RL agent can learn more effectively from some transitions than from others.

### Which Transitions to Prioritize?

The central component of prioritized replay is the criterion by which the importance of each transition is measured. Not all experiences are created equal; some contain vital lessons, while others are repetitive or uninformative.

Here are some possible criteria we can use to measure transition importance:

#### 1. Expected Learning Progress (The Ideal Criterion)
One idealized criterion would be the amount the RL agent can learn from a transition in its current state (**expected learning progress**).

#### 2. TD Error (The Practical Proxy)
While the ideal measure is not directly accessible, a reasonable proxy is the **magnitude of a transition’s TD error ($|\delta|$)**. This algorithm stores the last encountered TD error along with each transition in the replay memory. The transition with the largest absolute TD error ($|\delta|$) is replayed from the memory. A Q-learning update is applied to this transition, which updates the weights in proportion to the TD error.

> **The Catch:** The TD error can be a poor estimate in some circumstances as well, such as when environment transitions or rewards are highly noisy.

### Problems with Pure TD Error Prioritization

#### 1. Stale TD Errors & Starvation
To avoid expensive, computationally heavy sweeps over the entire replay memory pool, TD errors are only updated for the specific transitions that are actively replayed.

One major consequence of this is that transitions having a low TD error on their very first visit may **never or rarely be replayed again**, keeping their recorded error permanently low even if the evolving neural network could now learn a massive amount from them.

#### 2. High Sensitivity to Noise
Pure TD error prioritization is exceptionally sensitive to noise spikes. If an environment has stochastic transitions or noisy rewards, a transition might yield a massive TD error purely due to randomness rather than actual structural patterns. The agent will end up hyper-focusing on these uninformative, noisy anomalies.

We overcome some errors from TD error sampling by introducing Stochastic prioritization. This is a hybrid of TD error sampling and uniform random sampling. We ensure that the probability of being sampled is monotonic in a transition’s priority, while guaranteeing a non-zero probability even for the lowest-priority transition.

Probability of sampling transition $i$ as:

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

Depending on how we choose $p_i$ we can categorize this in many more categories. Two famous ones are:

#### Direct, proportional prioritization
where $p_i = |\delta_i| + \epsilon$, where $\delta$ is the TD error.

#### Indirect, rank-based prioritization
where $p_i = \frac{1}{\text{rank}(i)}$

Implementation of proportional variants is different, and also admits an efficient implementation based on a ‘sum-tree’ data structure (where every node is the sum of its children, with the priorities as the leaf nodes), which can be efficiently updated and sampled from.

### Algorithm: Double DQN with Proportional Prioritization

1. **Input:** minibatch $k$, step-size $\eta$, replay period $K$ and size $N$, exponents $\alpha$ and $\beta$, budget $T$.
2. **Initialize** replay memory $\mathcal{H} = \emptyset$, $\Delta = 0$, $p_1 = 1$
3. **Observe** $S_0$ and choose $A_0 \sim \pi_\theta(S_0)$
4. **for** $t = 1$ **to** $T$ **do**
5. &nbsp;&nbsp;&nbsp;&nbsp;**Observe** $S_t$, $R_t$, $\gamma_t$
6. &nbsp;&nbsp;&nbsp;&nbsp;**Store** transition $(S_{t-1}, A_{t-1}, R_t, \gamma_t, S_t)$ in $\mathcal{H}$ with maximal priority $p_t = \max_{i < t} p_i$
7. &nbsp;&nbsp;&nbsp;&nbsp;**if** $t \equiv 0 \pmod K$ **then**
8. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**for** $j = 1$ **to** $k$ **do**
9. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Sample** transition $j \sim P(j) = \frac{p_j^\alpha}{\sum_i p_i^\alpha}$
10. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Compute** importance-sampling weight $w_j = \frac{(N \cdot P(j))^{-\beta}}{\max_i w_i}$
11. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Compute** TD-error $\delta_j = R_j + \gamma_j Q_{\text{target}}\left(S_j, \arg\max_a Q(S_j, a)\right) - Q(S_{t-1}, A_{t-1})$
12. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Update** transition priority $p_j \leftarrow |\delta_j|$
13. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Accumulate** weight-change $\Delta \leftarrow \Delta + w_j \cdot \delta_j \cdot \nabla_\theta Q(S_{t-1}, A_{t-1})$
14. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end for**
15. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Update** weights $\theta \leftarrow \theta + \eta \cdot \Delta$, reset $\Delta = 0$
16. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;From time to time copy weights into target network $\theta_{\text{target}} \leftarrow \theta$
17. &nbsp;&nbsp;&nbsp;&nbsp;**end if**
18. &nbsp;&nbsp;&nbsp;&nbsp;**Choose** action $A_t \sim \pi_\theta(S_t)$
19. **end for**

Let’s not get in too much detail but line 10 corresponds to a concept called **“annealing the bias”**.

### Code for PER with Proportional Prioritization
The code below is a template to adapt the above logic into code.

```python
import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        # tree[0] is root; leaves occupy indices [capacity-1, 2*capacity-2]
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_index = 0
        self.size = 0

    def total(self):
        """Return the total sum of all priorities."""
        return self.tree[0]

    def add(self, priority, data):
        """Store a transition with the given priority."""
        leaf_index = self.write_index + self.capacity - 1
        self.data[self.write_index] = data
        self.update(leaf_index, priority)
        self.write_index = (self.write_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_index, priority):
        """Update the priority of a leaf node and propagate."""
        delta = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += delta

    def get(self, value):
        """Retrieve a leaf by sampling a value in [0, total())."""
        index = 0
        while index < self.capacity - 1:
            left = 2 * index + 1
            right = left + 1
            if value <= self.tree[left]:
                index = left
            else:
                value -= self.tree[left]
                index = right
        data_index = index - (self.capacity - 1)
        return index, self.tree[index], self.data[data_index]

class PrioritizedReplayBuffer:
    """Replay buffer that samples transitions proportional to their TD-error priority."""
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_steps=200_000, epsilon=1e-6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.epsilon = epsilon
        self.max_priority = 1.0
        self.step_count = 0

    def _get_beta(self):
        """Linearly anneal beta from beta_start to beta_end."""
        fraction = min(self.step_count / self.beta_steps, 1.0)
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def store(self, state, action, reward, next_state, done):
        """Store a transition with max priority."""
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, (state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch proportional to priorities."""
        indices = []
        priorities = []
        transitions = []
        segment = self.tree.total() / batch_size
        beta = self._get_beta()
        self.step_count += 1

        for i in range(batch_size):
            low, high = segment * i, segment * (i + 1)
            value = np.random.uniform(low, high)
            tree_idx, priority, data = self.tree.get(value)
            indices.append(tree_idx)
            priorities.append(priority)
            transitions.append(data)

        # Importance-sampling weights to correct for sampling bias
        priorities = np.array(priorities, dtype=np.float32)
        sampling_probs = priorities / self.tree.total()
        is_weights = (self.tree.size * sampling_probs) ** (-beta)
        is_weights /= is_weights.max() # normalise so max weight = 1

        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices, is_weights,
        )

    def update_priorities(self, indices, td_errors):
        """Update priorities after learning."""
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.size
```

## Dueling Networks
Standard deep Q-networks try to estimate the total value of taking a specific action in a specific state all at once. Dueling DQN optimizes this by splitting the network’s head into two separate streams: one that estimates the State-Value function $V(s)$ (how good it is to be in this state, regardless of action), and another that calculates the Advantage function $A(s,a)$ (how much better a specific action is compared to the state's average). By decoupling these two metrics, the network can learn which states are inherently valuable or dangerous without needing to obsessively learn the outcome of every single action available in that state.

## Noisy Nets
Classic DQN relies on a crude exploration strategy called $\varepsilon$-greedy, where the agent occasionally takes a completely random action based on a decaying probability threshold. Noisy Nets replace this erratic guessing game by adding parametric noise directly to the weights of the linear layers within the neural network itself. Because this noise is part of the network's parameters, the agent can naturally learn to reduce its own inner uncertainty over time. This gives the agent a much more stable, state-dependent mechanism for exploration, allowing it to systematically try new strategies in complex environments where random actions would fail.

## Distributional RL
Traditional reinforcement learning aims to model the expected return—the average reward an agent anticipates over the long run. Distributional RL radically shifts this paradigm by mapping the entire probability distribution of value returns ($Z$) rather than just a single mean number. Instead of asking, "What is my average expected reward?", a distributional framework tracks the full spectrum of potential outcomes, including risks and high-payoff anomalies. This gives the agent a massive performance boost, especially in highly unpredictable or chaotic environments, by providing a much richer, multi-faceted understanding of future rewards.

## N-step Learning
Standard Q-learning operates on a strict single-step update rule: the agent takes an action, observes a single immediate reward, and estimates the rest of the future using its current network weights. N-step learning expands this horizon by accumulating rewards over $n$ consecutive steps before looking at the network's bootstrap estimate. This simple adjustment strikes an elegant balance in the classic bias-variance tradeoff. By looking further down the road, the agent relies less on its own potentially inaccurate neural network approximations (reducing bias) while avoiding the instability of tracking an entire episode to the very end (controlling variance), resulting in vastly accelerated propagation of reward signals.

## The Grand Finale: Rainbow DQN
Individually, each extension we have discussed addresses a distinct, fundamental flaw in the original Deep Q-Network design. But a burning question remained for researchers: Are these independent patches, or can they be combined to build something truly definitive?

In 2017, researchers at DeepMind answered this by integrating all six extensions—Double DQN, Prioritized Experience Replay, Dueling Networks, Noisy Nets, Distributional RL, and N-step Learning—into a single agent named Rainbow DQN.

What makes Rainbow so elegant is that these components don't just sit alongside one another; they actively harmonize. For instance, Noisy Nets provide advanced exploration, completely replacing the need for classic $\varepsilon$-greedy exploration. Meanwhile, Prioritized Experience Replay measures priority using Kullback-Leibler (KL) divergence (a way to compare probability distributions) because Distributional RL has replaced single-value outputs with full reward distributions.

By accumulating rewards across several frames with N-step Learning and neutralizing maximization bias using a Double DQN target scheme, Rainbow achieved a massive leap in data efficiency and raw score metrics, establishing a classic milestone for state-of-the-art, value-based deep reinforcement learning.
