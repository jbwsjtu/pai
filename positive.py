import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
import pandas as pd
import os
import warnings

# ============================================================
# PyTorch for PPO-R baseline
# ============================================================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Bernoulli
except ImportError:
    raise ImportError("Please install PyTorch to run the PPO algorithm: pip install torch")

warnings.filterwarnings("ignore")


# ============================================================
# Global experiment config
# ============================================================
FULL_SWEEP = True



N = 70
M = 4
T = 1000
GAMMA = 0.99
NUM_SEEDS = 10

PPO_TRAIN_EPISODES = 50
PPO_TRAIN_STEPS = 100

MH_GRETA_H = 5


ADD_CUMULATIVE_MH_GRETA = False

# Environment parameters
P0, PA, PZ = 0.1, 0.1, 0.4
Q0, QA, QZ = 0.2, 0.1, 0.4
COST = 0.1

GRAPH_SEED = 42
GRAPH_M = 2

RESULT_DIR = "results"


# ============================================================
# 1. Environment
# ============================================================
class CanonicalDDERMAB:
    def __init__(self, N, M, P, alpha, beta,
                 p0, pa, pz, q0, qa, qz, cost, gamma, seed=None):
        self.N = N
        self.M = M
        self.P = P.astype(np.float64)
        self.alpha = alpha
        self.beta = beta
        self.p0, self.pa, self.pz = p0, pa, pz
        self.q0, self.qa, self.qz = q0, qa, qz
        self.cost = cost
        self.gamma = gamma
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.s = (self.rng.rand(self.N) < 0.3).astype(np.float32)
        self.z = np.zeros(self.N, dtype=np.float32)
        return self.s.copy(), self.z.copy()

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)

        reward = np.sum(self.s - self.cost * actions)

        injection = self.beta * actions
        self.z = self.alpha * (self.P @ self.z) + injection
        self.z = np.clip(self.z, 0, None)

        p_good = np.clip(self.p0 + self.pa * actions + self.pz * self.z, 0, 1)
        p_bad = np.clip(self.q0 - self.qa * actions - self.qz * self.z, 0, 1)

        mask_bad = (self.s == 0)
        mask_good = (self.s == 1)

        self.s[mask_bad] = (
            self.rng.rand(mask_bad.sum()) < p_good[mask_bad]
        ).astype(np.float32)

        self.s[mask_good] = 1 - (
            self.rng.rand(mask_good.sum()) < p_bad[mask_good]
        ).astype(np.float32)

        return (self.s.copy(), self.z.copy()), reward


# ============================================================
# 2. Single-arm DP and exposure sensitivity
# ============================================================
def solve_single_arm_value(p0, pa, pz, q0, qa, qz,
                           beta_local, cost, gamma, z_grid):
    n_z = len(z_grid)
    V = np.zeros((2, n_z))
    Q = np.zeros((2, n_z, 2))

    for _ in range(5000):
        V_old = V.copy()

        for s in [0, 1]:
            for iz, z in enumerate(z_grid):
                for a in [0, 1]:
                    if s == 0:
                        p_g = np.clip(p0 + pa * a + pz * z, 0, 1)
                        nxt_probs = [1 - p_g, p_g]
                    else:
                        p_b = np.clip(q0 - qa * a - qz * z, 0, 1)
                        nxt_probs = [p_b, 1 - p_b]

                    exp_V = nxt_probs[0] * V[0, iz] + nxt_probs[1] * V[1, iz]
                    Q[s, iz, a] = (s - cost * a) + gamma * exp_V

                V[s, iz] = np.max(Q[s, iz, :])

        if np.max(np.abs(V - V_old)) < 1e-6:
            break

    advantage = Q[:, :, 1] - Q[:, :, 0]
    return V, advantage


def compute_nu_analytical(env, V, z_grid):

    nu = np.zeros((2, len(z_grid)))

    for s in [0, 1]:
        for iz, z in enumerate(z_grid):
            dp_dz = env.pz if s == 0 else env.qz
            nu[s, iz] = env.gamma * dp_dz * (V[1, iz] - V[0, iz])

    return nu


# ============================================================
# 3. Policy definitions
# ============================================================
class MyopicPolicy:
    def act(self, s, z, env):
        actions = np.zeros(env.N, dtype=int)
        actions[np.argsort(s)[-env.M:]] = 1
        return actions


class TrueWhittlePolicy:
    def __init__(self, env, z_grid):
        env_zero = copy.deepcopy(env)
        env_zero.alpha = 0.0

        _, advantage = solve_single_arm_value(
            env_zero.p0, env_zero.pa, env_zero.pz,
            env_zero.q0, env_zero.qa, env_zero.qz,
            env_zero.beta, env_zero.cost, env_zero.gamma, z_grid
        )

        self.whittle_index = advantage[:, 0]

    def act(self, s, z, env):
        scores = self.whittle_index[s.astype(int)]

        actions = np.zeros(env.N, dtype=int)
        actions[np.argsort(scores)[-env.M:]] = 1
        return actions


class GRETAPolicy:
    """
    GRETA-style one-hop graph-aware baseline.

    """

    def __init__(self, env, z_grid):
        self.env = env
        self.z_grid = z_grid

        env_zero = copy.deepcopy(env)
        env_zero.alpha = 0.0

        V, advantage = solve_single_arm_value(
            env_zero.p0, env_zero.pa, env_zero.pz,
            env_zero.q0, env_zero.qa, env_zero.qz,
            env_zero.beta, env_zero.cost, env_zero.gamma, z_grid
        )

        self.V = V
        self.advantage = advantage

    def _lookup(self, table, s_vals, z_vals):
        idx = np.clip(
            np.searchsorted(self.z_grid, z_vals),
            1,
            len(self.z_grid) - 1
        )
        z_l, z_r = self.z_grid[idx - 1], self.z_grid[idx]
        frac = (z_vals - z_l) / (z_r - z_l + 1e-12)

        return (
            (1 - frac) * table[s_vals.astype(int), idx - 1]
            + frac * table[s_vals.astype(int), idx]
        )

    def act(self, s, z, env):
        scores = np.zeros(env.N)
        v_current = self._lookup(self.V, s, z)

        for u in range(env.N):
            # 1. Local activation advantage
            w_pull = self._lookup(
                self.advantage,
                np.array([s[u]]),
                np.array([z[u]])
            )[0]

            
            z_new = z + env.beta * env.P[:, u]
            v_next = self._lookup(self.V, s, z_new)

            w_msg_total = np.sum(v_next - v_current) - (v_next[u] - v_current[u])

            scores[u] = w_pull + w_msg_total

        actions = np.zeros(env.N, dtype=int)
        actions[np.argsort(scores)[-env.M:]] = 1
        return actions


class MultiHopGRETAPolicy:
    """
    Multi-hop GRETA-style baseline.
    """

    def __init__(self, env, z_grid, H=5, exclude_source=True,
                 use_discount=True, cumulative=False):
        self.env = env
        self.z_grid = z_grid
        self.H = H
        self.exclude_source = exclude_source
        self.use_discount = use_discount
        self.cumulative = cumulative

        # Consistent with GRETA: use the decoupled local value backbone with alpha=0.
        env_zero = copy.deepcopy(env)
        env_zero.alpha = 0.0

        V, advantage = solve_single_arm_value(
            env_zero.p0, env_zero.pa, env_zero.pz,
            env_zero.q0, env_zero.qa, env_zero.qz,
            env_zero.beta, env_zero.cost, env_zero.gamma, z_grid
        )

        self.V = V
        self.advantage = advantage

    def _lookup(self, table, s_vals, z_vals):
        idx = np.clip(
            np.searchsorted(self.z_grid, z_vals),
            1,
            len(self.z_grid) - 1
        )
        z_l, z_r = self.z_grid[idx - 1], self.z_grid[idx]
        frac = (z_vals - z_l) / (z_r - z_l + 1e-12)

        return (
            (1 - frac) * table[s_vals.astype(int), idx - 1]
            + frac * table[s_vals.astype(int), idx]
        )

    def act(self, s, z, env):
        scores = np.zeros(env.N)
        v_current = self._lookup(self.V, s, z)

        for u in range(env.N):
            # 1. Local activation advantage, same as GRETA.
            w_pull = self._lookup(
                self.advantage,
                np.array([s[u]]),
                np.array([z[u]])
            )[0]

            msg_total = 0.0

        
            hop_signal = env.beta * env.P[:, u].copy()

            if self.cumulative:
                cumulative_signal = np.zeros(env.N)

            for h in range(1, self.H + 1):
                if np.max(np.abs(hop_signal)) < 1e-12:
                    break

                if self.cumulative:
                    cumulative_signal = cumulative_signal + hop_signal
                    z_new = z + cumulative_signal
                else:
                    z_new = z + hop_signal

                v_next = self._lookup(self.V, s, z_new)
                gains = v_next - v_current

             
                if self.exclude_source:
                    gains[u] = 0.0

                # Use gamma^(h-1) so that the weight is 1 when H=1, strictly matching GRETA.
                weight = env.gamma ** (h - 1) if self.use_discount else 1.0
                msg_total += weight * np.sum(gains)

                # Next delayed hop.
                hop_signal = env.alpha * (env.P @ hop_signal)

            scores[u] = w_pull + msg_total

        actions = np.zeros(env.N, dtype=int)
        actions[np.argsort(scores)[-env.M:]] = 1
        return actions


class PAIPolicy:
   

    def __init__(self, env, z_grid):
        self.env, self.z_grid = env, z_grid

        V, advantage = solve_single_arm_value(
            env.p0, env.pa, env.pz,
            env.q0, env.qa, env.qz,
            env.beta, env.cost, env.gamma, z_grid
        )

        self.advantage = advantage
        self.nu_table = compute_nu_analytical(env, V, z_grid)

        I = np.eye(env.N)
        self.resolvent = np.linalg.inv(I - env.gamma * env.alpha * env.P.T)

    def _lookup(self, table, s_vals, z_vals):
        idx = np.clip(
            np.searchsorted(self.z_grid, z_vals),
            1,
            len(self.z_grid) - 1
        )
        z_l, z_r = self.z_grid[idx - 1], self.z_grid[idx]
        frac = (z_vals - z_l) / (z_r - z_l + 1e-12)

        return (
            (1 - frac) * table[s_vals.astype(int), idx - 1]
            + frac * table[s_vals.astype(int), idx]
        )

    def act(self, s, z, env):
        delta = self._lookup(self.advantage, s, z)
        nu = self._lookup(self.nu_table, s, z)

     
        psi_inf = env.beta * (self.resolvent @ nu)
        scores = delta + psi_inf

        actions = np.zeros(env.N, dtype=int)
        actions[np.argsort(scores)[-env.M:]] = 1
        return actions


# ============================================================
# 4. PPO-R baseline
# ============================================================
class SharedActorCritic(nn.Module):
    def __init__(self, state_dim=2):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)


class PPOPolicy_AAAI25:
    """
    PPO-R baseline.

    This is a learning-based RMAB baseline:
    - shared actor-critic across arms;
    - local input: (s_i, z_i);
    - test time selects top-M arms by activation probability.
    """

    def __init__(self, env, num_train_episodes=30, max_steps=100, torch_seed=0):
        self.env = env
        self.device = torch.device("cpu")

        torch.manual_seed(torch_seed)
        np.random.seed(torch_seed)

        self.model = SharedActorCritic(state_dim=2).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-3)

        self.train_ppo(num_train_episodes, max_steps)

    def train_ppo(self, episodes, max_steps):
        gamma = self.env.gamma
        clip_ratio = 0.2
        ppo_epochs = 4

        train_env = copy.deepcopy(self.env)

        for ep in range(episodes):
            s, z = train_env.reset()

            states_buf = []
            actions_buf = []
            rewards_buf = []
            values_buf = []
            logprobs_buf = []

            for step in range(max_steps):
                state_tensor = torch.tensor(
                    np.stack([s, z], axis=1),
                    dtype=torch.float32
                ).to(self.device)

                with torch.no_grad():
                    probs, values = self.model(state_tensor)
                    probs = probs.squeeze(-1)
                    probs = torch.clamp(probs, 1e-5, 1.0 - 1e-5)
                    values = values.squeeze(-1)

                probs_np = probs.cpu().numpy()

                actions = np.zeros(train_env.N, dtype=np.float32)
                actions[np.argsort(probs_np)[-train_env.M:]] = 1.0

                (s_next, z_next), _ = train_env.step(actions)

                indiv_rewards = s - train_env.cost * actions

                states_buf.append(state_tensor)
                actions_buf.append(
                    torch.tensor(actions, dtype=torch.float32).to(self.device)
                )
                rewards_buf.append(
                    torch.tensor(indiv_rewards, dtype=torch.float32).to(self.device)
                )
                values_buf.append(values)

                dist = Bernoulli(probs)
                logp = dist.log_prob(
                    torch.tensor(actions, dtype=torch.float32).to(self.device)
                )
                logprobs_buf.append(logp)

                s, z = s_next, z_next

            states_t = torch.cat(states_buf)
            actions_t = torch.cat(actions_buf)
            old_logp_t = torch.cat(logprobs_buf)

            rewards_mat = torch.stack(rewards_buf)
            values_mat = torch.stack(values_buf)

            returns_mat = torch.zeros_like(rewards_mat)

            state_last = torch.tensor(
                np.stack([s, z], axis=1),
                dtype=torch.float32
            ).to(self.device)

            with torch.no_grad():
                _, last_val = self.model(state_last)
                running_ret = last_val.squeeze(-1)

            for t in reversed(range(max_steps)):
                running_ret = rewards_mat[t] + gamma * running_ret
                returns_mat[t] = running_ret

            adv_mat = returns_mat - values_mat
            adv_mat = (adv_mat - adv_mat.mean()) / (adv_mat.std() + 1e-8)

            returns_t = returns_mat.reshape(-1)
            adv_t = adv_mat.reshape(-1)

            for _ in range(ppo_epochs):
                probs_new, values_new = self.model(states_t)
                probs_new = probs_new.squeeze(-1)
                probs_new = torch.clamp(probs_new, 1e-5, 1.0 - 1e-5)
                values_new = values_new.squeeze(-1)

                dist_new = Bernoulli(probs_new)
                logp_new = dist_new.log_prob(actions_t)

                ratio = torch.exp(logp_new - old_logp_t)
                clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_t

                loss_pi = -(torch.min(ratio * adv_t, clip_adv)).mean()
                loss_v = nn.MSELoss()(values_new, returns_t)

                loss = loss_pi + 0.5 * loss_v

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def act(self, s, z, env):
        state_tensor = torch.tensor(
            np.stack([s, z], axis=1),
            dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            probs, _ = self.model(state_tensor)
            probs = probs.squeeze(-1).cpu().numpy()

        actions = np.zeros(env.N, dtype=int)
        actions[np.argsort(probs)[-env.M:]] = 1
        return actions


# ============================================================
# 5. Simulation utilities
# ============================================================
def run_simulation(env, policy, T=200, seed=42):
    env.rng.seed(seed)
    s, z = env.reset()

    rewards = []

    for _ in range(T):
        actions = policy.act(s, z, env)
        (s, z), r = env.step(actions)
        rewards.append(r)

    rewards = np.array(rewards)
    discounts = env.gamma ** np.arange(T)

    return np.sum(discounts * rewards)


def run_action_overlap(env, policy_a, policy_b, T=200, seed=42):
   
    env.rng.seed(seed)
    s, z = env.reset()

    same_count = 0
    jaccards = []

    for _ in range(T):
        a1 = policy_a.act(s, z, env)
        a2 = policy_b.act(s, z, env)

        set1 = set(np.where(a1 == 1)[0])
        set2 = set(np.where(a2 == 1)[0])

        if set1 == set2:
            same_count += 1

        inter = len(set1.intersection(set2))
        union = len(set1.union(set2))

        if union == 0:
            jaccards.append(1.0)
        else:
            jaccards.append(inter / union)

        # Just advance the environment using policy_a.
        (s, z), _ = env.step(a1)

    same_rate = same_count / T
    mean_jaccard = float(np.mean(jaccards))

    return same_rate, mean_jaccard


def build_graph_and_propagation(N, graph_m=2, seed=42):
    G = nx.barabasi_albert_graph(N, graph_m, seed=seed)
    A = nx.adjacency_matrix(G).toarray().astype(float)

    # Add self-loops
    np.fill_diagonal(A, 1.0)

    row_sums = A.sum(axis=1, keepdims=True)
    P = A / row_sums

    return G, P


def make_env(P, alpha, beta, seed=None):
    return CanonicalDDERMAB(
        N=N,
        M=M,
        P=P,
        alpha=alpha,
        beta=beta,
        p0=P0,
        pa=PA,
        pz=PZ,
        q0=Q0,
        qa=QA,
        qz=QZ,
        cost=COST,
        gamma=GAMMA,
        seed=seed
    )


# ============================================================
# 6. Main experiment
# ============================================================
def run_optimized_sweep():
    os.makedirs(RESULT_DIR, exist_ok=True)

    if FULL_SWEEP:
        alpha_list = [0.2, 0.4, 0.6, 0.8]
        beta_list = [0.4, 1.0, 1.2, 1.4]
    else:
        # Fast rebuttal version
        alpha_list = [0.6, 0.8]
        beta_list = [1.0, 1.2]

    alpha_list = [0.2, 0.4, 0.6, 0.8]
    beta_list = [0.4, 1.0, 1.2, 1.4]

    alpha_list = [0.6]
    beta_list = [1.2]


    _, P = build_graph_and_propagation(N, graph_m=GRAPH_M, seed=GRAPH_SEED)

    policy_names = [
        "Myopic",
        "WhittleIndex",
        "PPO-R",
        "GRETA",
        "MH-GRETA",
        "PAI (Ours)"
    ]

    if ADD_CUMULATIVE_MH_GRETA:
        policy_names.insert(-1, "MH-GRETA-Cum")

    print("=" * 90)
    print("Starting evaluation: Myopic / WhittleIndex / PPO-R / GRETA / MH-GRETA / PAI")
   
    print("=" * 90)

    results_db = {}
    rows = []
    overlap_rows = []

    for beta in beta_list:
        for alpha in alpha_list:
            print(f"\n[Test combination] Alpha = {alpha}, Beta = {beta}")

            env_base = make_env(P, alpha, beta, seed=0)

            z_max = beta / (1 - alpha + 1e-6) * 2
            z_grid = np.linspace(0, z_max, 50)

            policies = {
                "Myopic": MyopicPolicy(),
                "WhittleIndex": TrueWhittlePolicy(env_base, z_grid),
                "PPO-R": PPOPolicy_AAAI25(
                    env_base,
                    num_train_episodes=PPO_TRAIN_EPISODES,
                    max_steps=PPO_TRAIN_STEPS,
                    torch_seed=0
                ),
                "GRETA": GRETAPolicy(env_base, z_grid),
                "MH-GRETA": MultiHopGRETAPolicy(
                    env_base,
                    z_grid,
                    H=MH_GRETA_H,
                    exclude_source=True,
                    use_discount=True,
                    cumulative=False
                ),
                "PAI (Ours)": PAIPolicy(env_base, z_grid)
            }

            if ADD_CUMULATIVE_MH_GRETA:
                policies["MH-GRETA-Cum"] = MultiHopGRETAPolicy(
                    env_base,
                    z_grid,
                    H=MH_GRETA_H,
                    exclude_source=True,
                    use_discount=True,
                    cumulative=True
                )

            # ------------------------------------------------------------
            # Action overlap diagnostics
            # ------------------------------------------------------------
            overlap_env = make_env(P, alpha, beta, seed=123)

            same_rate, jaccard = run_action_overlap(
                overlap_env,
                policies["GRETA"],
                policies["MH-GRETA"],
                T=200,
                seed=123
            )

            print(
                f"   [Diagnostic] GRETA vs MH-GRETA action same rate = {same_rate:.3f}, "
                f"Jaccard = {jaccard:.3f}"
            )

            overlap_rows.append({
                "alpha": alpha,
                "beta": beta,
                "policy_a": "GRETA",
                "policy_b": "MH-GRETA",
                "same_rate": same_rate,
                "mean_jaccard": jaccard,
                "diagnostic_T": 200,
                "MH_GRETA_H": MH_GRETA_H
            })

            if ADD_CUMULATIVE_MH_GRETA:
                overlap_env_2 = make_env(P, alpha, beta, seed=123)

                same_rate_cum, jaccard_cum = run_action_overlap(
                    overlap_env_2,
                    policies["GRETA"],
                    policies["MH-GRETA-Cum"],
                    T=200,
                    seed=123
                )

                print(
                    f"   [Diagnostic] GRETA vs MH-GRETA-Cum action same rate = {same_rate_cum:.3f}, "
                    f"Jaccard = {jaccard_cum:.3f}"
                )

                overlap_rows.append({
                    "alpha": alpha,
                    "beta": beta,
                    "policy_a": "GRETA",
                    "policy_b": "MH-GRETA-Cum",
                    "same_rate": same_rate_cum,
                    "mean_jaccard": jaccard_cum,
                    "diagnostic_T": 200,
                    "MH_GRETA_H": MH_GRETA_H
                })

            # ------------------------------------------------------------
            # Evaluation
            # ------------------------------------------------------------
            results_db[(alpha, beta)] = {}

            for name in policy_names:
                seed_returns = []

                for seed in range(NUM_SEEDS):
                    eval_env = make_env(P, alpha, beta, seed=seed)

                    ret = run_simulation(
                        eval_env,
                        policies[name],
                        T=T,
                        seed=seed
                    )

                    seed_returns.append(ret)

                seed_returns = np.array(seed_returns)

                mean_ret = float(np.mean(seed_returns))
                std_ret = float(np.std(seed_returns))
                sem_ret = float(std_ret / np.sqrt(NUM_SEEDS))

                results_db[(alpha, beta)][name] = {
                    "mean": mean_ret,
                    "std": std_ret,
                    "sem": sem_ret
                }

                rows.append({
                    "alpha": alpha,
                    "beta": beta,
                    "policy": name,
                    "mean": mean_ret,
                    "std": std_ret,
                    "sem": sem_ret,
                    "num_seeds": NUM_SEEDS,
                    "N": N,
                    "M": M,
                    "T": T,
                    "gamma": GAMMA,
                    "MH_GRETA_H": MH_GRETA_H
                })

                print(f"   => {name:15s}: {mean_ret:8.2f}  ± {std_ret:6.2f}")

    # ========================================================
    # Save results
    # ========================================================
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULT_DIR, "multihop_baseline_results.csv")
    df.to_csv(csv_path, index=False)

    overlap_df = pd.DataFrame(overlap_rows)
    overlap_csv_path = os.path.join(RESULT_DIR, "action_overlap_diagnostics.csv")
    overlap_df.to_csv(overlap_csv_path, index=False)

    print("\n" + "=" * 90)
    print(f"Main results saved to: {csv_path}")
    print(f"Action overlap diagnostics saved to: {overlap_csv_path}")
    print("=" * 90)

    print("\nMain results DataFrame:")
    print(df)

    print("\nAction overlap diagnostics:")
    print(overlap_df)

    # ========================================================
    # Plot grouped bar chart
    # ========================================================
    plot_results(results_db, alpha_list, beta_list, policy_names)


def plot_results(results_db, alpha_list, beta_list, policy_names):
    os.makedirs(RESULT_DIR, exist_ok=True)

    n_beta = len(beta_list)

    fig, axes = plt.subplots(
        1,
        n_beta,
        figsize=(5.3 * n_beta, 4.8),
        sharey=True
    )

    if n_beta == 1:
        axes = [axes]

    color_map = {
        "Myopic": "#9467bd",
        "WhittleIndex": "#ff7f0e",
        "PPO-R": "#1f77b4",
        "GRETA": "#2ca02c",
        "MH-GRETA": "#8c564b",
        "MH-GRETA-Cum": "#e377c2",
        "PAI (Ours)": "#d62728"
    }

    for idx, beta in enumerate(beta_list):
        ax = axes[idx]

        x_positions = np.arange(len(alpha_list))
        total_width = 0.84
        n_policies = len(policy_names)
        width = total_width / n_policies

        for i, name in enumerate(policy_names):
            color = color_map.get(name, None)

            means = [
                results_db[(alpha, beta)][name]["mean"]
                for alpha in alpha_list
            ]

            sems = [
                results_db[(alpha, beta)][name]["sem"]
                for alpha in alpha_list
            ]

            offset = (i - n_policies / 2 + 0.5) * width

            ax.bar(
                x_positions + offset,
                means,
                width,
                yerr=sems,
                label=name,
                color=color,
                capsize=3,
                alpha=0.9,
                edgecolor="black",
                linewidth=1
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"$\\alpha={a}$" for a in alpha_list], fontsize=11)
        ax.set_title(f"Injection $\\beta={beta}$", fontsize=13, fontweight="bold")
        ax.set_xlabel("Memory ($\\alpha$)", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        if idx == 0:
            ax.set_ylabel("Discounted Return", fontsize=12)

        if idx == 0:
            ax.legend(fontsize=8, loc="upper left")

    plt.suptitle(
        f"PAI vs Baselines with Multi-hop GRETA-H{MH_GRETA_H}",
        fontsize=16,
        y=1.05,
        fontweight="bold"
    )

    plt.tight_layout()

    fig_path = os.path.join(RESULT_DIR, "final_comparison_with_mh_greta.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")

    print(f"\nFigure saved to: {fig_path}")

    plt.show()


if __name__ == "__main__":
    run_optimized_sweep()
