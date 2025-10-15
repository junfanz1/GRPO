"""
Search-R1 (GRPO Edition)
=========================
Reinforcement learning training loop based on GRPO (Group Relative Policy Optimization)
for training an Agentic RAG / Search-Enhanced Language Model.

Key Features:
- Combines search actions with answer generation actions
- Uses GRPO (group-relative PPO) instead of standard policy gradients
- No critic network required
- Compares multiple candidate trajectories
- KL regularization to stabilize training
- Detailed annotations aligned with Search-R1 / DeepSeek-R1 experimental design
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import List


# ============================================================
# 🧱 Data Structures
# ============================================================

@dataclass
class TokenStep:
    """Record of a single token generation"""
    token_id: int
    token_text: str
    log_prob: float
    position: int  # Position in the full sequence (used for recomputing log probs)


@dataclass
class Trajectory:
    """Complete trajectory (including question, search, and answer generation)"""
    question: str
    answer: str
    token_steps: List[TokenStep]
    generated_text: str  # Model-generated text including intermediate search commands
    full_input_ids: List[int]
    generated_positions: List[int]
    reward: float = 0.0  # Reward value (computed later)


# ============================================================
# 🔍 Search Engine (replaceable with real RAG)
# ============================================================

class SearchEngine:
    """Simple local knowledge base search engine"""

    def __init__(self):
        self.knowledge_base = {
            "machine learning": "Machine learning is a subset of AI that enables computers to learn from experience.",
            "neural networks": "Neural networks are computing systems inspired by biological neural networks.",
            "deep learning": "Deep learning is a subset of machine learning using artificial neural networks.",
            "transformer": "Transformers are neural network architectures using self-attention mechanisms.",
            "reinforcement learning": "Reinforcement learning involves agents learning through environment interaction.",
        }

    def search(self, query: str) -> str:
        """Return a simple search result"""
        query_lower = (query or "").lower().strip()
        for key, value in self.knowledge_base.items():
            if key in query_lower:
                return value
        return f"No information found for: {query}"


# ============================================================
# 🧩 Search-R1 Trainer (GRPO-based)
# ============================================================

class SearchR1Trainer:
    def __init__(self, model_name="Qwen/Qwen2.5-7B", device=None):
        # Load language model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    # --------------------------------------------------------
    # 🧠 Trajectory Generation (agent interaction)
    # --------------------------------------------------------
    def generate_trajectory(self, question: str, search_engine: SearchEngine) -> Trajectory:
        """
        Simulate agent answering process: may perform searches (<search> ... </search>) before generating answers.
        """
        self.model.eval()
        current_text = question
        input_ids = self.tokenizer.encode(question, return_tensors="pt").to(self.device)

        token_steps = []
        full_input_ids = input_ids[0].tolist()
        generated_positions = []

        max_steps = 100
        done = False

        for _ in range(max_steps):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)

            # Sample next token from probability distribution
            next_token_id = torch.multinomial(probs, num_samples=1).squeeze()
            log_prob = torch.log(probs[0, next_token_id])
            token_text = self.tokenizer.decode([next_token_id.item()], skip_special_tokens=True)

            # === Fix position indexing ===
            future_pos = len(full_input_ids)
            generated_positions.append(future_pos)

            # Record token generation
            token_steps.append(TokenStep(
                token_id=next_token_id.item(),
                token_text=token_text,
                log_prob=log_prob.item(),
                position=future_pos,
            ))

            # Update input sequence
            full_input_ids.append(next_token_id.item())
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)

            # Append to current text
            current_text += token_text

            # Check if entering search mode
            if "<search>" in current_text and "</search>" in current_text:
                query = self.extract_search_query(current_text)
                search_result = search_engine.search(query)
                # Inject search result into context
                context_addition = f"\n[Search result]: {search_result}\n"
                current_text += context_addition
                search_tokens = self.tokenizer.encode(context_addition, add_special_tokens=False)
                full_input_ids.extend(search_tokens)
                input_ids = torch.tensor([full_input_ids], device=self.device)

            # Termination condition
            if any(end in current_text.lower() for end in ["</answer>", "<eos>", "end of answer"]):
                done = True
                break

        return Trajectory(
            question=question,
            answer=current_text,
            token_steps=token_steps,
            generated_text=current_text,
            full_input_ids=full_input_ids,
            generated_positions=generated_positions,
        )

    # --------------------------------------------------------
    def extract_search_query(self, text: str) -> str:
        """Extract query between <search> ... </search>"""
        start_tag, end_tag = "<search>", "</search>"
        start = text.find(start_tag) + len(start_tag)
        end = text.find(end_tag)
        if start >= len(start_tag) and end > start:
            return text[start:end].strip()
        return ""

    # --------------------------------------------------------
    # 🎯 Reward Function (replaceable with semantic similarity)
    # --------------------------------------------------------
    def compute_reward(self, prediction: str, ground_truth: str) -> float:
        """
        Simple reward: 1 if prediction contains ground_truth, else 0.
        Can replace with embedding similarity or LLM evaluation.
        """
        return 1.0 if ground_truth.strip().lower() in prediction.strip().lower() else 0.0

    # --------------------------------------------------------
    # 🔁 Recompute trajectory log_probs (for KL and new policy)
    # --------------------------------------------------------
    def recompute_log_probs(self, trajectories: List[Trajectory]) -> List[torch.Tensor]:
        self.model.eval()

        input_ids_list, attention_masks, adjusted_positions = [], [], []
        for traj in trajectories:
            input_ids_tensor = torch.tensor(traj.full_input_ids, dtype=torch.long, device=self.device)
            attention_mask = torch.ones_like(input_ids_tensor, device=self.device)
            input_ids_list.append(input_ids_tensor)
            attention_masks.append(attention_mask)
            adjusted_positions.append(traj.generated_positions)

        max_len = max(len(ids) for ids in input_ids_list)
        input_ids_padded = torch.stack([
            F.pad(ids, (0, max_len - len(ids)), value=self.tokenizer.pad_token_id)
            for ids in input_ids_list
        ])
        attention_mask_padded = torch.stack([
            F.pad(mask, (0, max_len - len(mask)), value=0)
            for mask in attention_masks
        ])

        with torch.no_grad():
            outputs = self.model(input_ids_padded, attention_mask=attention_mask_padded)
            logits = outputs.logits  # [batch, seq_len, vocab]

        all_log_probs = []
        for i, (traj, positions) in enumerate(zip(trajectories, adjusted_positions)):
            log_probs = []
            for pos, token_step in zip(positions, traj.token_steps):
                pred_index = max(pos - 1, 0)  # causal LM correction
                log_prob_tensor = F.log_softmax(logits[i, pred_index], dim=-1)[token_step.token_id]
                log_probs.append(log_prob_tensor)
            all_log_probs.append(torch.stack(log_probs))
        return all_log_probs

    # --------------------------------------------------------
    # 🧮 KL Divergence (stable implementation)
    # --------------------------------------------------------
    def compute_kl_divergence(self, old_log_probs: torch.Tensor, new_log_probs: torch.Tensor) -> torch.Tensor:
        """KL(old || new) = sum(p_old * (log_old - log_new))"""
        p_old = torch.exp(old_log_probs)
        kl = torch.sum(p_old * (old_log_probs - new_log_probs))
        return kl

    # --------------------------------------------------------
    # 🧩 GRPO Update Step (core)
    # --------------------------------------------------------
    def update_policy(self, trajectories: List[Trajectory], beta: float = 0.01) -> torch.Tensor:
        """
        Compute GRPO loss:
            - Compute group mean reward
            - Compute relative advantage
            - Add KL regularization
        """
        rewards = torch.tensor([t.reward for t in trajectories], dtype=torch.float32, device=self.device)
        mean_reward = rewards.mean()
        advantages = rewards - mean_reward  # Group-relative advantage

        # Recompute new policy log_probs
        new_log_probs_list = self.recompute_log_probs(trajectories)
        old_log_probs_list = [
            torch.tensor([step.log_prob for step in traj.token_steps], dtype=torch.float32, device=self.device).detach()
            for traj in trajectories
        ]

        policy_losses, kl_losses = [], []
        for adv, old_lp, new_lp in zip(advantages, old_log_probs_list, new_log_probs_list):
            seq_len = min(len(old_lp), len(new_lp))
            old_lp, new_lp = old_lp[:seq_len], new_lp[:seq_len]
            kl = self.compute_kl_divergence(old_lp, new_lp)
            kl_losses.append(kl)
            policy_losses.append(-adv * torch.sum(new_lp))

        policy_loss = torch.stack(policy_losses).mean()
        kl_loss = torch.stack(kl_losses).mean()
        total_loss = policy_loss + beta * kl_loss

        return total_loss

    # --------------------------------------------------------
    # 🚀 Single Training Step (GRPO)
    # --------------------------------------------------------
    def train_step(self, queries, answers, search_engine, optimizer, num_candidates=4, beta=0.01):
        """
        Perform GRPO update for a batch:
        - Generate num_candidates trajectories per query
        - Compute group-relative reward
        - Compute loss and backpropagate
        """
        self.model.train()
        batch_trajectories = []

        for question, ground_truth in zip(queries, answers):
            group_trajs = []
            for _ in range(num_candidates):
                traj = self.generate_trajectory(question, search_engine)
                traj.reward = self.compute_reward(traj.generated_text, ground_truth)
                group_trajs.append(traj)

            batch_trajectories.extend(group_trajs)

        loss = self.update_policy(batch_trajectories, beta=beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    # --------------------------------------------------------
    # 🔁 Full Training Loop
    # --------------------------------------------------------
    def train(self, dataset, search_engine, epochs=3, lr=1e-5, num_candidates=4, beta=0.01):
        optimizer = Adam(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            total_loss = 0.0
            for question, answer in dataset:
                loss = self.train_step([question], [answer], search_engine, optimizer, num_candidates, beta)
                total_loss += loss
            print(f"[Epoch {epoch+1}] Avg Loss = {total_loss / len(dataset):.4f}")


# ============================================================
# ✅ Example Run
# ============================================================

if __name__ == "__main__":
    # Simple dataset
    dataset = [
        ("What is deep learning?", "Deep learning is a subset of machine learning."),
    ]

    search_engine = SearchEngine()
    trainer = SearchR1Trainer(model_name="Qwen/Qwen2.5-7B")  # can swap for smaller model for testing
    trainer.train(dataset, search_engine, epochs=2, lr=1e-5, num_candidates=3, beta=0.02)
