# Reinforcement Learning for Agentic Search in LLMs

Search-R1 fine-tunes LLMs to decide when to search and when to answer using reinforcement learning over multi-step trajectories. It employs Group Relative Policy Optimization (GRPO) for stable token-level updates without a critic. The model learns adaptive retrieval policies, dynamically integrating search results to construct context and improve reasoning accuracy.

### — A Deep Dive into Search-R1 and Reward-Guided Reasoning

> *“What if LLMs could learn **when to search** and **when to answer** — just like humans refining their thoughts through Google?”*

---

## 1. Motivation: Why Search-R1?

The recent **Search-R1** paradigm—popularized by researchers exploring retrieval-augmented agents—extends the traditional RAG pipeline with **reinforcement learning**.

Instead of statically retrieving information, the model **learns a search policy** that decides *when* and *how* to query external knowledge sources.

> The model doesn’t just use context; it *learns to orchestrate* its own context.

This moves us from **RAG (retrieval-augmented generation)** → **Agentic RAG (retrieval-augmented reasoning)**.

---

## 2. Why Not Simple Policy Gradient?

A naive REINFORCE policy gradient works, but it’s unstable:

* Treats each sample independently.
* Does not handle reward variance properly.
* Ignores relative quality of responses in a batch.

Enter **GRPO — Group Relative Policy Optimization**, proposed in *DeepSeek-R1* and used in Search-R1.

---

## 3. What is GRPO?

GRPO is an RL objective designed to stabilize learning **without a critic model**.

Instead of comparing an action’s reward to a value function baseline, GRPO compares **each action’s reward to the mean reward of its group**:

\[
L_{GRPO} = - \mathbb{E}_{(x, a, r) \sim D} \Big[ \frac{r - \bar{r}_{group}}{\sigma_{group}} \cdot \log \pi_\theta(a|x) \Big]
\]

Where:

* \(r\): reward of one trajectory  
* \(\bar{r}_{group}\): mean reward of trajectories in the same batch  
* \(\sigma_{group}\): standard deviation of group rewards  

This **relative normalization** creates a self-contained baseline, stabilizing training even for small batch RL fine-tuning of LLMs.

---

## 4. Architecture Overview

```
User Question → Model → Output Trajectory
↓
Search or Answer?
↓
(if <search>) Query Search Engine
↓
Append Search Results to Context
↓
Final Answer → Compute Reward(Answer, Ground Truth)
↓
GRPO Update: Adjust Model via Relative Reward Gradients
```


The model learns a **search-aware reasoning loop** through reward feedback.

---

## 5. Full GRPO Training Implementation (PyTorch)

```python
import torch
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

# 1. Model and Tokenizer Setup
model_name = "Qwen/Qwen2.5-7B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Dummy Search Engine
class SearchEngine:
    def search(self, query):
        # Can integrate Serper.dev, Bing API, or local vector DB
        return f"[search results for: {query}]"
search_engine = SearchEngine()

# 3. Trajectory Generation
def generate_trajectory(model, tokenizer, question, search_engine, max_steps=3):
    trajectory, log_probs, actions = [], [], []
    state = question
    done = False
    for step in range(max_steps):
        inputs = tokenizer(state, return_tensors="pt")
        outputs = model.generate(
            **inputs, max_new_tokens=100,
            output_scores=True, return_dict_in_generate=True
        )
        text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        last_logits = outputs.scores[-1].squeeze(0)
        probs = F.log_softmax(last_logits, dim=-1)
        trajectory.append(text)
        if "<search>" in text:
            actions.append("search")
            log_probs.append(probs.max())
            query = extract_search_query(text)
            search_results = search_engine.search(query)
            state += "\n" + search_results
        else:
            actions.append("answer")
            log_probs.append(probs.max())
            done = True
            break
    return trajectory, torch.stack(log_probs), actions

def extract_search_query(text):
    start_tag, end_tag = "<search>", "</search>"
    start = text.find(start_tag) + len(start_tag)
    end = text.find(end_tag)
    return text[start:end].strip() if start >= len(start_tag) and end > start else ""

# 4. Reward Function
def compute_reward(prediction, ground_truth):
    return torch.tensor(1.0 if prediction.strip() == ground_truth.strip() else 0.0)

# 5. GRPO Loss
def compute_grpo_loss(log_probs_batch, rewards_batch):
    rewards = torch.stack(rewards_batch)
    log_probs = torch.stack([lp.mean() for lp in log_probs_batch])
    mean_r = rewards.mean()
    std_r = rewards.std() + 1e-8
    normed_r = (rewards - mean_r) / std_r
    loss = -torch.mean(normed_r * log_probs)
    return loss

# 6. GRPO Training Loop
def train_grpo(model, tokenizer, dataset, search_engine, epochs=3, lr=1e-5):
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        log_probs_batch, rewards_batch = [], []
        for question, ground_truth in dataset:
            trajectory, log_probs, actions = generate_trajectory(model, tokenizer, question, search_engine)
            reward = compute_reward(trajectory[-1], ground_truth)
            log_probs_batch.append(log_probs)
            rewards_batch.append(reward)
        loss = compute_grpo_loss(log_probs_batch, rewards_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
    return model
```

## 6. GRPO vs PPO / REINFORCE

| Feature           | REINFORCE     | PPO                    | GRPO              |
| ----------------- | ------------- | ---------------------- | ----------------- |
| Baseline          | None / manual | Value model            | Group mean reward |
| Stability         | Low           | High (requires critic) | High (no critic)  |
| Batch sensitivity | Unstable      | Stable                 | Stable            |
| Use case          | Toy RL        | RLHF                   | R1 / Search-R1    |

> GRPO can be seen as a **Critic-free PPO**, making it ideal for LLM fine-tuning where computing value estimates is difficult or expensive.

---

## 7. Conceptual Implication: Learning to Search

The GRPO + Search-R1 setup enables the LLM to:

* **Learn when to search** intelligently using `<search>` tags.
* **Minimize unnecessary queries** for trivial questions.
* **Internalize context engineering**, structuring its own evidence for reasoning.

Through iterative RL fine-tuning, the model transitions from:

> “Generate then search” → “Search only when needed” → “Integrate retrieval as part of reasoning.”

---

## 8. Future Directions

1. **Multi-Step GRPO**
   * Extend reward to full trajectory for multi-hop reasoning.
2. **LLM-as-a-Judge Rewards**
   * Replace binary reward with semantic or GPT-based scoring for reasoning quality.
3. **Hierarchical GRPO**
   * Separate policies for `search` vs `reason`, forming modular sub-agents.
4. **Memory-Augmented Search**
   * Combine GRPO with vector memory replay (e.g., Mem0, memories.ai) to recall prior searches.

---

## 9. Final Takeaway

> GRPO transforms LLMs into **autonomous information-seeking agents**.  
> Moves from context-fed models → context-constructing agents, and from prompt engineering → reinforcement-driven context engineering.

* Enables **search-aware reasoning loops** at token-level.
* Stabilizes RL fine-tuning without requiring a critic.
* Encourages **intelligent retrieval and reasoning integration**.

---

## 10. TL;DR

| Concept        | Description                                               |
| -------------- | --------------------------------------------------------- |
| **Goal**       | Train an LLM to decide when/how to search                 |
| **Algorithm**  | GRPO (Group Relative Policy Optimization)                 |
| **Mechanism**  | Compare rewards within batch to derive relative advantage |
| **Outcome**    | Stable RL fine-tuning without critic                      |
| **Philosophy** | Teach LLMs *how to seek knowledge*, not just *recite it*  |

---

## 11. Optional Diagram: Search-Reasoning Loop

```text
[User Query]
      ↓
   [LLM Agent]
      ↓
+-----------------+
| Decide Action   |
| <search> or <answer>
+-----------------+
      ↓
  [Search Engine]
      ↓
[Integrate Info]
      ↓
 [Generate Answer]
      ↓
 [Compute Reward] → GRPO Update
```




# 🔍 Search-R1: Reinforcement Fine-Tuning for Agentic Retrieval-Augmented Generation (RAG)

**Search-R1** is a minimal yet fully functional **reinforcement learning framework** that teaches a large language model *when to search and when to answer* — implementing **Agentic RAG** behavior through **Group Relative Policy Optimization (GRPO)**.

---

## 🚀 Overview

Traditional RAG pipelines are **static** — they always search before answering.  
**Search-R1** introduces **adaptive retrieval**, where the model dynamically decides:

- whether additional search is needed  
- how to formulate the query  
- and when to stop searching and produce the final answer  

The model learns this **search–reason–answer** cycle via **reinforcement learning**, using **GRPO** as a stable policy optimization objective.

---

## 🧠 Core Idea

> Combine **language modeling** with **policy optimization** to train an *agentic retriever* that learns to optimize reasoning through active information seeking.

Each episode (one question) becomes an RL trajectory:
1. The model observes the question (state).
2. It generates actions:
   - `<search> query </search>` → triggers a search operation.
   - `<answer> ... </answer>` → final answer.
3. The search engine executes the query and appends results to the context.
4. The model continues generating until an answer is given.
5. GRPO computes reward-based gradients over trajectory groups for stable training.

---

## 🧩 Architecture

```
┌──────────────────────────────────────────┐
│ Question │
│ ↓ │
│ [LLM] → "<search> query </search>" │
│ ↓ │
│ [Search Engine] → search results │
│ ↓ │
│ [LLM] → reasoning + "<answer> ..." │
└──────────────────────────────────────────┘
```


Each episode forms a **search–reason–answer loop**,  
where the model *learns when and how to retrieve* to maximize accuracy with minimal search cost.

---

## ⚙️ Training Pipeline

| Step | Description |
|------|--------------|
| **1. Trajectory Generation** | Model generates `<search>` and `<answer>` tokens given a question. |
| **2. Environment Interaction** | `<search>` triggers external retrieval (e.g., API, vector DB, or web). |
| **3. Reward Computation** | Reward = correctness / similarity between model answer and ground truth. |
| **4. GRPO Optimization** | Use group-wise baselines to reduce variance and stabilize updates. |

---

## 🧮 GRPO Objective (Simplified)

For a batch of trajectories \( G = \{τ_i\} \):

\[
L_{GRPO} = - \mathbb{E}_{τ_i \in G} [(R_i - \bar{R}_G) \cdot \log \pi_θ(a_i|s_i)]
\]

- \( R_i \): reward for trajectory \( τ_i \)  
- \( \bar{R}_G \): average reward across group \( G \) (acts as a baseline)  
- \( \pi_θ \): model policy (token-level distribution)  

Compared to vanilla policy gradient:
- ✅ **Variance reduction** (group baseline)  
- ✅ **Stability** across heterogeneous reasoning trajectories  
- ✅ **Improved convergence** with small sample sizes  

Optional enhancements:
- Entropy regularization: encourages exploration  
- Gradient clipping: prevents instability  
- Reward normalization: keeps updates scale-consistent  

---

## 🧰 Reward Design

| Type | Description | Implementation |
|------|--------------|----------------|
| **Exact Match** | `1.0` if answer matches ground truth, else `0.0` | Simple QA or closed-domain tasks |
| **Semantic Similarity** | `cosine(emb(pred), emb(gt))` | Use Sentence Transformers / OpenAI Embeddings |
| **LLM-as-a-Judge** | Score(answer, reference) ∈ [0, 1] | Use GPT-4 or Qwen2.5-Judge model for grading |

Composite reward example:
```python
reward = 0.7 * semantic_similarity + 0.3 * llm_judge_score
```

## 🧩 Code Structure

```
search-r1/
│
├── search_r1_train.py      # main RL training loop (GRPO)
├── trajectory.py           # trajectory generation + search integration
├── rewards.py              # reward computation (exact / semantic / LLM-judge)
├── search_engine.py        # abstract + local/web retrieval interface
├── models/                 # model loading, LoRA fine-tuning utilities
│   └── qwen_wrapper.py
│
├── notebooks/
│   ├── search_r1_demo.ipynb   # Colab-ready demo
│   └── reward_analysis.ipynb  # visualize reward distribution
│
├── requirements.txt
└── README.md
```

## 🧪 Example Usage

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
from search_r1_train import train_grpo
from search_engine import LocalSearch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

search_engine = LocalSearch(index_dir="./data")
dataset = [("Who discovered penicillin?", "Alexander Fleming")]

trained_model = train_grpo(model, tokenizer, dataset, search_engine, epochs=3, lr=1e-5)
```
---

## ⚡ Implementation Highlights

- 🧠 **Dynamic Action Space**:  
  The model learns *when* to trigger `<search>` tokens versus `<answer>`, effectively treating each generated token as an RL action.  
- 🧩 **End-to-End Trainable**:  
  Trajectories (state, action, log probability, reward) are used to backpropagate gradients through **GRPO**, enabling token-level policy updates.  
- 🧮 **Group Relative Policy Optimization (GRPO)**:  
  - Reduces variance by normalizing rewards within trajectory groups.  
  - Stable convergence even with heterogeneous multi-step reasoning trajectories.  
  - Probability ratio clipping (`1 ± ε`) and KL regularization ensure updates are smooth.  
- 🪶 **PEFT / LoRA Ready**:  
  Supports parameter-efficient fine-tuning to reduce GPU memory footprint while training large LLMs.  
- 🔍 **Pluggable Search API**:  
  Any search/retrieval backend (FAISS, local DB, web API) can be integrated via `search_engine.search(query)`.  
- 🧾 **Trajectory Logging & Replay**:  
  Stores token-level actions, log probabilities, rewards, and full input sequences for analysis or replay buffers.  
- ⚖️ **Reward Flexibility**:  
  - Exact match reward  
  - Semantic similarity (embedding-based)  
  - LLM-as-a-judge scoring  
  Can be combined or weighted to form custom reward functions.  
- 🔒 **Stability Techniques**:  
  - Gradient clipping  
  - Advantage normalization  
  - KL divergence regularization to avoid catastrophic policy updates  

---

## 🧠 Research Context

This project builds on state-of-the-art ideas in **retrieval-augmented reasoning agents**:

- **DeepSeek-R1**: Reinforcement fine-tuning of reasoning LLMs  
- **Search-Augmented Agents (OpenAI, 2025)**: Adaptive search policies in multi-step reasoning  
- **GRPO**: Group-relative policy optimization for low-variance token-level updates  
- **Context Engineering & Memory Systems**: Aligning retrieved knowledge with reasoning trajectory  

Search-R1 demonstrates how **neural reasoning** can be combined with **symbolic retrieval actions** in an RL framework, forming a new class of agentic LLMs.

---

## 🧭 Design Principles

| Principle | Description |
|------------|--------------|
| **Reason over Retrieve** | Retrieval is optional; the model learns when it is necessary. |
| **Learnable Search Policy** | The model internalizes optimal retrieval strategies through GRPO updates. |
| **Reward Alignment** | Encourages factually correct and well-formatted answers. |
| **Compositional Reasoning** | Combines search, information integration, and answer generation in one trajectory. |
| **Stable Training** | Variance reduction via group baselines, clipping, and KL regularization. |

---

## 📈 Future Extensions

- 🔁 **Trajectory Replay Buffer**: Sample multiple episodes to improve policy stability  
- 🧩 **Hierarchical Rewards**: Multi-step reasoning with intermediate reward signals  
- 🧮 **RLAIF Integration**: Human-in-the-loop reward alignment for alignment-focused tasks  
- 🧠 **Multi-Agent Retrieval**: Collaborative agents (Coordinator + Researcher) for complex reasoning  
- 🌐 **Web/Vector Search Expansion**: Extend beyond local knowledge bases to large-scale RAG environments  

---

## 🌌 TL;DR

**Search-R1 = (RAG + Reasoning) × Reinforcement Learning**

> Train language models not just to *answer*,  
> but to *search, think, and verify* autonomously, learning optimal retrieval strategies with GRPO.

---

## 📜 References

- [Hands on LLM CN, Chaofa Yuan](https://github.com/bbruceyuan/Hands-On-Large-Language-Models-CN)
