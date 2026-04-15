import numpy as np
import random
import pickle
import time
from collections import deque, namedtuple
from typing import List, Optional, Tuple

Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "done", "mask"]
)


#  Neural Network Components 

class Linear:
    def __init__(self, in_dim: int, out_dim: int, seed: int = 0):
        rng        = np.random.default_rng(seed)
        scale      = np.sqrt(2.0 / in_dim)
        self.W     = rng.standard_normal((in_dim, out_dim)) * scale
        self.b     = np.zeros(out_dim)
        self.dW    = np.zeros_like(self.W)
        self.db    = np.zeros_like(self.b)

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self._x.T @ grad
        self.db = grad.sum(axis=0)
        return grad @ self.W.T

    @property
    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad):
        return grad * self._mask


class MLP:
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 num_heads: int, seed: int = 42):
        self.num_heads = num_heads
        self.l1   = Linear(in_dim, hidden, seed)
        self.r1   = ReLU()
        self.l2   = Linear(hidden, hidden, seed + 1)
        self.r2   = ReLU()
        self.heads = [Linear(hidden, out_dim, seed + 2 + i) for i in range(num_heads)]

    def forward(self, x):
        h = self.r2.forward(self.l2.forward(self.r1.forward(self.l1.forward(x))))
        self._h = h
        return [head.forward(h) for head in self.heads]

    def predict(self, x):
        h = np.maximum(0, x @ self.l1.W + self.l1.b)
        h = np.maximum(0, h @ self.l2.W + self.l2.b)
        return [h @ head.W + head.b for head in self.heads]

    def getWeights(self):
        layers = [self.l1, self.l2] + self.heads
        return [(np.copy(l.W), np.copy(l.b)) for l in layers]

    def setWeights(self, weights):
        layers = [self.l1, self.l2] + self.heads
        for layer, (W, b) in zip(layers, weights):
            layer.W[:] = W
            layer.b[:] = b

    def paramCount(self) -> int:
        return sum(w.size + b.size for w, b in self.getWeights())


class Adam:
    def __init__(self, lr=5e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.lr  = lr
        self.b1  = b1
        self.b2  = b2
        self.eps = eps
        self.t   = 0
        self.m   = {}
        self.v   = {}

    def step(self, params):
        self.t += 1
        for i, (p, g) in enumerate(params):
            if i not in self.m:
                self.m[i] = np.zeros_like(p)
                self.v[i] = np.zeros_like(p)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g ** 2
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def huberLoss(pred, target, delta=1.0) -> Tuple[float, np.ndarray]:
    err = pred - target
    ae  = np.abs(err)
    loss = np.where(ae <= delta, 0.5 * err ** 2, delta * (ae - 0.5 * delta))
    grad = np.where(ae <= delta, err, delta * np.sign(err))
    return float(loss.mean()), grad / max(len(pred), 1)


#  Replay Memory 

class ReplayMemory:
    def __init__(self, capacity: int, num_heads: int):
        self.capacity  = capacity
        self.num_heads = num_heads
        self.buffer    = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        mask = np.random.randint(0, 2, size=self.num_heads).astype(np.float32)
        self.buffer.append(Transition(state, action, reward, next_state, done, mask))

    def sample(self, n: int) -> List[Transition]:
        return random.sample(self.buffer, n)

    def __len__(self):
        return len(self.buffer)

    @property
    def is_ready(self) -> bool:
        return len(self) >= self.capacity // 50


#  Bootstrapped DQN Agent 

class BootstrappedDQNAgent:
    """
    BDQL++ — Bootstrapped Deep Q-Learning for clinical trial arm assignment.

    State  : patient pharmacogenomic profile (encoded gene metabolizer statuses)
    Action : trial arm index (drug selection)
    Reward : simulated outcome grounded in PharmaGKB gene-drug associations
    """

    def __init__(self, state_dim: int, action_dim: int, drug_names: List[str], cfg: dict):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.drug_names = drug_names
        self.cfg        = cfg
        self.num_heads  = cfg["num_heads"]

        self.online = MLP(state_dim, cfg["hidden_size"], action_dim, self.num_heads)
        self.target = MLP(state_dim, cfg["hidden_size"], action_dim, self.num_heads)
        self.target.setWeights(self.online.getWeights())

        self.optim  = Adam(lr=cfg["learning_rate"])
        self.memory = ReplayMemory(cfg["memory_size"], self.num_heads)

        self.epsilon      = cfg["epsilon_start"]
        self.eps_end      = cfg["epsilon_end"]
        self.eps_decay    = cfg["epsilon_decay"]
        self.active_head  = 0
        self.update_count = 0
        self.steps_done   = 0

        # Training history
        self.episode_rewards : List[float] = []
        self.episode_losses  : List[float] = []
        self.epsilons        : List[float] = []
        self.arm_counts      : List[int]   = [0] * action_dim
        self.head_certainty  : List[float] = []

    #  Action selection 
    def selectAction(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        q_all = self.online.predict(state[np.newaxis])
        if greedy:
            avg_q = np.mean(q_all, axis=0).flatten()
            return int(np.argmax(avg_q))
        return int(np.argmax(q_all[self.active_head].flatten()))

    def get_action_confidence(self, state: np.ndarray) -> dict:
        """Return per-arm confidence scores and head votes."""
        q_all   = self.online.predict(state[np.newaxis])      # list of K (1,A) arrays
        q_stack = np.vstack([q.flatten() for q in q_all])     # (K, A)

        avg_q   = q_stack.mean(axis=0)
        std_q   = q_stack.std(axis=0)
        votes   = np.argmax(q_stack, axis=1)                   # which arm each head prefers

        # Softmax to confidence %
        shifted = avg_q - avg_q.max()
        exp_q   = np.exp(shifted)
        probs   = exp_q / exp_q.sum()

        head_votes = {self.drug_names[a]: int((votes == a).sum()) for a in range(self.action_dim)}

        return {
            "probabilities" : probs.tolist(),
            "avg_q"         : avg_q.tolist(),
            "std_q"         : std_q.tolist(),
            "head_votes"    : head_votes,
            "best_action"   : int(np.argmax(avg_q)),
            "certainty"     : float(1.0 - std_q[np.argmax(avg_q)] / (np.abs(avg_q).mean() + 1e-6)),
        }

    def rotate_head(self):
        self.active_head = random.randrange(self.num_heads)

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    #  Training step 
    def train_step(self) -> Optional[float]:
        if not self.memory.is_ready:
            return None

        batch      = self.memory.sample(self.cfg["batch_size"])
        states     = np.array([t.state      for t in batch], dtype=np.float32)
        actions    = np.array([t.action     for t in batch], dtype=np.int32)
        rewards    = np.array([t.reward     for t in batch], dtype=np.float32)
        nexts      = np.array([t.next_state for t in batch], dtype=np.float32)
        dones      = np.array([t.done       for t in batch], dtype=np.float32)
        masks      = np.array([t.mask       for t in batch], dtype=np.float32)   # (B, K)

        gamma       = self.cfg["gamma"]
        next_q_all  = self.target.predict(nexts)
        online_q_all= self.online.forward(states)

        total_grads = [np.zeros_like(q) for q in online_q_all]
        total_loss  = 0.0
        all_params  = []

        for k in range(self.num_heads):
            hm = masks[:, k]
            if hm.sum() == 0:
                continue
            tq        = online_q_all[k].copy()
            best_next = next_q_all[k].max(axis=1)
            td_target = rewards + gamma * best_next * (1.0 - dones)
            for i in range(len(batch)):
                if hm[i] > 0:
                    tq[i, actions[i]] = td_target[i]
            lk, gk = huberLoss(online_q_all[k], tq)
            gk *= hm[:, np.newaxis]
            total_grads[k] += gk
            total_loss += lk * hm.mean()

        # Backward through heads then shared encoder
        shared_grad = np.zeros((len(batch), self.online.l2.W.shape[1]))
        for k, head in enumerate(self.online.heads):
            g = head.backward(total_grads[k])
            shared_grad += g
            all_params.extend(head.params)

        g = self.online.r2.backward(shared_grad)
        g = self.online.l2.backward(g)
        all_params.extend(self.online.l2.params)
        g = self.online.r1.backward(g)
        g = self.online.l1.backward(g)
        all_params.extend(self.online.l1.params)

        self.optim.step(all_params)

        self.update_count += 1
        if self.update_count % self.cfg["target_update_freq"] == 0:
            self.target.setWeights(self.online.getWeights())

        return total_loss / max(self.num_heads, 1)

    #  Checkpoint 
    def save(self, path: str):
        cp = {
            "online": self.online.getWeights(),
            "target": self.target.getWeights(),
            "epsilon": self.epsilon,
            "steps": self.steps_done,
            "rewards": self.episode_rewards,
            "losses": self.episode_losses,
            "arm_counts": self.arm_counts,
            "cfg": self.cfg,
            "drug_names": self.drug_names,
        }
        with open(path, "wb") as f:
            pickle.dump(cp, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            cp = pickle.load(f)
        self.online.setWeights(cp["online"])
        self.target.setWeights(cp["target"])
        self.epsilon         = cp["epsilon"]
        self.steps_done      = cp["steps"]
        self.episode_rewards = cp["rewards"]
        self.episode_losses  = cp["losses"]
        self.arm_counts      = cp.get("arm_counts", self.arm_counts)

    #  Quantization (int8 approximation) 
    def quantize_weights(self) -> dict:
        """Simulate int8 quantization and return size comparison."""
        weights = self.online.getWeights()
        original_bytes = sum(w.nbytes + b.nbytes for w, b in weights)

        quantized = []
        for w, b in weights:
            scale_w = np.abs(w).max() / 127.0 + 1e-8
            scale_b = np.abs(b).max() / 127.0 + 1e-8
            qw = np.clip(np.round(w / scale_w), -127, 127).astype(np.int8)
            qb = np.clip(np.round(b / scale_b), -127, 127).astype(np.int8)
            quantized.append((qw, qb, scale_w, scale_b))

        quantized_bytes = sum(qw.nbytes + qb.nbytes for qw, qb, _, _ in quantized)

        return {
            "original_kb"  : original_bytes / 1024,
            "quantized_kb" : quantized_bytes / 1024,
            "compression"  : original_bytes / quantized_bytes,
            "params"       : self.online.paramCount(),
        }

    def prune_to_best_head(self) -> "BootstrappedDQNAgent":
        """Extract the single best-performing head as a lightweight agent."""
        q_vals = []
        rng = np.random.default_rng(42)
        test_states = rng.standard_normal((50, self.state_dim)).astype(np.float32)
        for k in range(self.num_heads):
            q = np.array([self.online.predict(s[np.newaxis])[k].max() for s in test_states])
            q_vals.append(q.mean())
        best_k = int(np.argmax(q_vals))

        small_cfg = {**self.cfg, "num_heads": 1}
        small = BootstrappedDQNAgent(self.state_dim, self.action_dim, self.drug_names, small_cfg)
        # Copy shared layers + best head
        full_weights = self.online.getWeights()
        small_weights = [full_weights[0], full_weights[1], full_weights[2 + best_k]]
        small.online.setWeights(small_weights)
        small.target.setWeights(small_weights)
        return small, best_k
