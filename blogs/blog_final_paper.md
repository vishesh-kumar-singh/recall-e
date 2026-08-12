# RECALL-E: Continual Reinforcement Learning via Online World Models

**Authors:** Chaitanya, Nishant, Adrija

## 1. Environment Frameworks: Continual vs. Unified World

### Continual World
In the standard Continual World framework, an agent tackles tasks sequentially. However, typical models work with an inconsistent state space across tasks. Because they learn task-by-task rather than mapping out a single, unified set of world dynamics, the network undergoes **catastrophic forgetting** when shifting across distinct task boundaries.

### Unified World (Continual Bench)
In a Unified World framework—such as **Continual Bench**—tasks are arranged spatially rather than strictly temporally. 
* **Global Physics:** All tasks share a single, consistent state space and a unified set of world dynamics ($P^u$). 
* **Zero Forgetting:** Instead of fitting isolated habits task-by-task, the agent learns a global representation of physics. Because the physics of the environment do not alter, the agent updates incrementally and remains immune to catastrophic forgetting.

---

## 2. Agenda & Reward System
The primary objective is to maximize the cumulative long-term return across all sequentially presented tasks without losing historical proficiencies. 
* **Follow-The-Leader (FTL):** The agent employs an FTL online learning scheme to build a highly stable world model.
* **Task-Driven Distributional Shifts:** While the underlying physics remain static, the reward function ($R^\tau$) varies over time according to the active task primitive. The agent naturally shifts its real-world trajectory to maximize the currently active reward, inducing a severe distributional data shift that standard networks cannot tolerate.

---

## 3. Planning, Shift-Initialization, and MPC

Action selection is handled via a derivative-free stochastic optimization loop paired with Model Predictive Control (MPC):



### The Optimization Process
1. **Gaussian Trajectory Distribution:** The planner initializes a candidate distribution using a mean action sequence ($\mu$) and an initial variance matrix ($\sigma$) over a fixed planning horizon $H$.
2. **Stochastic Sampling ($N$):** The system generates $N$ candidate action sequences. It incorporates **temporal colored noise** to promote smooth, continuous physical exploration rather than chaotic adjustments.
3. **Mental Rollouts:** Each action candidate sequence is simulated inside the agent's learned world model ($W$) to project future states and compute their expected rewards.
4. **Elite Fitting:** The top $10\%$ highest-rewarding sequences are isolated. The mean and variance are re-fitted to these elite paths, narrowing down the Gaussian curve toward a highly optimal trajectory. This refinement loop is executed over $K$ internal iterations.
5. **Model Predictive Control (MPC):** The robot executes only the *very first action* ($a_t$) of the optimized sequence in the real world, discarding the remaining $H-1$ steps.
6. **Shift-Initialization:** Rather than recomputing the planning distribution from absolute scratch on the next step, the planner takes the unused $H-1$ steps from the previous plan, shifts them forward by one step, and appends a baseline guess at the end. This significantly accelerates convergence during real-time tracking.

---

## 4. Online Agent Architecture & Sparsity

To accommodate rapid real-time updates without backpropagation, the Online Agent (OA) utilizes a wide, shallow architecture driven by linear algebra:



### Wide Shallow Network & Structural Isolation
The network consists of a fixed, high-dimensional random projection layer followed by a single trainable linear layer ($W$). Because the input is parsed into an exceptionally wide feature space, different physical task states trigger entirely localized, separate clusters of neurons. Adjusting weights for a new task alters parameters only within an active sub-block, keeping historical parameters safely locked down.

### The FTL Mathematical Engine
Instead of keeping a raw history of experiences, the agent tracks its lifetime progress continuously via two cumulative matrices:
* **Matrix $A$ ($D \times D$):** Accumulates the feature-to-feature correlation summaries over time.
* **Matrix $B$ ($D \times S$):** Accumulates the feature-to-target output correlations.
* **Vector $y_t$:** Tracks the true environmental state delta ($s_{t+1} - s_t$) produced by an action.

$$W_{s}^{(t)}=(A_{ss}^{(t-1)}+\frac{1}{\lambda}I)^{-1}(B_{s}^{(t-1)}-A_{s\overline{s}}^{(t-1)}W_{\overline{s}}^{(t-1)})$$

### Computation Bounds via Sparsity
Because the encoder forces a top-$K$ selection mechanism where $K \ll D$, the vast majority of the network stays completely quiet. When solving the analytical weight step, the model slices out submatrices ($A_{ss}$, $B_s$) linked exclusively to the active coordinates ($s$). This keeps matrix inversion highly localized and bounds step-by-step processing costs.

---

## 5. Metrics & Analytical Mechanics

### Regret
Regret mathematically captures the online performance penalty of the system. It is calculated as the integrated area representing the performance gap between a perfect Oracle model (which achieves a constant $1.0$ success rate) and the live Online Agent. Theorem 1 proves that the Online Agent achieves a sublinear regret bound, proving that error growth slows down and performance stabilizes.

### Soft Binning
Inputs mapped between $0$ and $1$ via a sigmoid function are projected onto a grid containing $\Lambda$ bins. The encoder uses a soft-binning function to find the lower and upper integer boundary edges enclosing the coordinate:
$$s_i = [\lfloor I_i \rfloor, \quad \lfloor I_i \rfloor + 1]$$
The dimensions of the total feature matrix scale according to the number of random features ($d$) and the bin width ($\Lambda$) such that $D = d \times \Lambda$. Higher values of $\Lambda$ broaden the total hidden layer, giving the network wider parameter capacity.

---

## 6. Baseline Performance Comparison










| Model / Approach | Plasticity (New Learning) | Stability (Retention) | Regret Metric Evaluation |
| :--- | :--- | :--- | :--- |
| **Fine-Tuning** | Highly Adaptive | Immediate Catastrophic Forgetting | Extremely High Regret Accumulation |
| **Synaptic Intelligence (SI)** | Moderate Plasticity | Alleviates forgetting marginally on early steps | High Regret; Struggles with complex shifts |
| **Coreset (Replay)** | Adaptive | Degrades as old sample representation drops | Moderate Regret; limited by buffer size |
| **Perfect Memory** | High upfront cost | Absolute retention via exhaustive replay | Low Regret but computationally impractical |
| **Online Agent (OA)** | Instant Convergence | Absolute retention via structural isolation | **Lowest Regret** with fixed constant overhead |

### Performance Analysis
* **Fine-Tuning:** The model learns the first task effectively, but as it sequentially encounters further tasks, it suffers from catastrophic forgetting, causing performance on earlier tasks to collapse.
* **Online Agent (OA):** By training incrementally, the agent maintains a success rate equivalent to the "Perfect Memory" baseline, effectively countering catastrophic forgetting without storing all historical data.
* **SI & Coreset:** These approaches show reasonable success on the initial two tasks, but their performance diminishes significantly as the sequence of tasks grows, as they cannot maintain stability over long-term continual learning.



### Sparse Model Dynamics
* **Neuron Utilization:** As the agent interacts with sequential tasks, network space utilization gradually grows. Due to the exact closed-form summarization, even if the model reaches near-maximum parameter utilization, it safely tracks an overall least-squares solution without overwriting old parameters.
* **Performance Scaling:** The model demonstrates that performance scales positively with increased utilization and a higher number of soft bins ($\Lambda$), providing the agent with more capacity to map complex world dynamics.



---

## 7. Hyperparameter Configurations

To establish an optimal sparse world model layout, the paper applies the following default hyperparameters:

* **Number of Projections ($d$):** 300 Losse Features
* **Bin Count ($\Lambda$):** 9 bins (swept across 5, 7, 9, 11)
* **Regularization Weight ($\frac{1}{\lambda}$):** 0.005
* **Planning Horizon ($H$):** 15 steps
* **CEM Refinement Iterations ($K$):** 3 iterations
* **Sample Paths ($N$):** 150 candidates
* **Elite Cutoff Ratio:** 0.1 ($10\%$)

---

## 8. Unified Training & Interaction Flow

```mermaid
flowchart TD
    Start(["Start: Agent Lifelong Loop (A.2)"]) --> RecvState["Receive Current State s_t"]
    RecvState --> NewTask{"Is it a new task /<br/>task change? (A.2)"}
    NewTask -->|YES| ResetMu["Reset μ_t to<br/>baseline initials"]
    NewTask -->|NO| ShiftInit["Apply Shift-Initialization<br/>to μ_t"]
    ResetMu --> WorldModel
    ShiftInit --> WorldModel

    WorldModel["Line 7: Closed-Form World Model Update (A.2)<br/>• Identify active feature indices (s)<br/>• Analytically solve W_s weights instantly"]
    WorldModel --> CEMEnter[/"ENTER PLANNED POLICY GENERATION: CEM (A.4)"/]

    subgraph CEM["CEM Loop"]
        CEMEnter --> InitSigma["Initialize σ_init"]
        InitSigma --> LoopStart["Loop k = 1 to K"]
        LoopStart --> Sample["Sample N=150 Action Path Candidates<br/>• Centered at μ, with temporal Colored Noise<br/>• Carry over elite memory samples from k-1 (if k>1)"]
        Sample --> Rollout["Run Mental Rollouts in World Model<br/>• Predict state shifts using learned weights W<br/>• Evaluate simulated returns via Reward Function R"]
        Rollout --> Elite["Identify Top 10% Elite Candidates<br/>• Fit a new Gaussian distribution to winners<br/>• Update μ and σ parameters for next generation"]
        Elite --> KCheck{"Does k equal K?"}
        KCheck -->|NO| LoopStart
        KCheck -->|YES| Output["Output Final Optimized<br/>Matrix μ_(t+1)"]
    end

    Output --> CEMExit[/"EXIT PLANNED POLICY GENERATION: CEM"/]
    CEMExit --> Execute["Execute First Action a_t<br/>in Real World (A.2)"]
    Execute --> NewState["Arrive at New State s_(t+1)"]
    NewState --> Delta["Calculate Experience Delta Summary (A.2)<br/>• Input: x_t = [s_t, a_t]<br/>• Outcome target: y_t = s_(t+1) - s_t"]
    Delta --> SparseProj["Line 11: Sparse Feature Projection (A.2)<br/>• Pass x_t through Random Matrix P and Sigmoid<br/>• Run Soft-Binning using grid size Λ<br/>• Isolate nonzero activated index positions (s)"]
    SparseProj --> IncUpdate["Lines 12-13: Incremental Matrix Updates (A.2)<br/>• Add feature correlation directly to Matrix A_ss<br/>• Add feature-target correlation directly to Matrix B_s"]
    IncUpdate --> IncStep["Increment Step: t ← t + 1"]
    IncStep -.->|"loop back to next step"| RecvState