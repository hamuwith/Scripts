import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from collections import deque
import numpy as np
from ChemicalChainEnvOnehot import ChemicalEnvOnehot 
from torch.utils.tensorboard import SummaryWriter
import os
from collections import Counter

class NStepBuffer:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.buffer = deque()

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def is_ready(self):
        return len(self.buffer) >= self.n

    def get(self):
        #nステップ分の報酬を合算して1つの遷移を返す
        R = 0
        for i in range(self.n):
            R += (self.gamma ** i) * self.buffer[i][2]  # reward

        state, action = self.buffer[0][:2]
        next_state, done = self.buffer[-1][3], self.buffer[-1][4]

        return state, action, R, next_state, done

    def pop(self):
        self.buffer.popleft()

    def clear(self):
        self.buffer.clear()

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity # 葉の数
        self.tree = np.zeros(2 * capacity - 1)  # 完全二分木のノード数
        self.data = np.zeros(capacity, dtype=object) # データを格納する配列
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]
    
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 重みとバイアスの平均と標準偏差
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)    
            
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
    
class SumTreePrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 1e-5

    def push(self, state, action, reward, next_state, done):
        max_priority = np.max(self.tree.tree[-self.capacity:])  # 葉の最大値
        if max_priority == 0:
            max_priority = 1.0

        data = (state, action, reward, next_state, done)
        self.tree.add(max_priority, data)

    def sample(self, batch_size, beta=0.4):
        total_priority = self.tree.total()
        if total_priority == 0:
            return None
        batch = []
        indices = []
        priorities = []

        segment = total_priority / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(p)

        probs = np.array(priorities) / total_priority
        n_entries = np.count_nonzero(self.tree.tree[-self.capacity:])
        weights = (n_entries * probs) ** (-beta)
        # weights = (len(self.tree.data) * probs) ** (-beta)
        weights /= weights.max()
        batch = list(zip(*batch))

        states = torch.stack(batch[0])
        actions = torch.tensor(batch[1], dtype=torch.long)
        rewards = torch.tensor(batch[2], dtype=torch.float32)
        next_states = torch.stack(batch[3])
        dones = torch.tensor(batch[4], dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            p = (abs(err) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)

    def __len__(self):
        return len([x for x in self.tree.data if x is not None])
    
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha  # 0: uniform, 1: full prioritization

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        data = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data

        self.priorities[self.pos] = max_prio  # 新しいデータは高い優先度
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # IS weight計算（重み補正）
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # 正規化

        batch = list(zip(*samples))
        states = np.array(batch[0])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.array(batch[3])
        dones = np.array(batch[4])
        weights = np.array(weights, dtype=np.float32)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors, eps=1e-5):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + eps        

    def __len__(self):
        return len(self.buffer)
            
class CategoricalDuelingNoisyNStepQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, atom_size=101, Vmin=-80, Vmax=1800, sigma_init= 0.017):
        super().__init__()
        self.action_dim = action_dim
        self.atom_size = atom_size
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.support = torch.linspace(self.Vmin, self.Vmax, self.atom_size)
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # 状態価値（V）を出力するネットワーク
        self.value = nn.Sequential(            
            NoisyLinear(hidden_dim, hidden_dim, sigma_init=sigma_init),
            nn.ReLU(),
            NoisyLinear(hidden_dim, hidden_dim, sigma_init=sigma_init),
            nn.ReLU(),
            NoisyLinear(hidden_dim, atom_size, sigma_init=sigma_init)
        )

        # アクション優位性（A）を出力するネットワーク
        self.advantage = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, sigma_init=sigma_init),
            nn.ReLU(),
            NoisyLinear(hidden_dim, hidden_dim, sigma_init=sigma_init),
            nn.ReLU(),
            NoisyLinear(hidden_dim, action_dim * atom_size, sigma_init=sigma_init)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature(x)
        adv = self.advantage(x).view(batch_size, self.action_dim, self.atom_size)
        val = self.value(x).view(batch_size, 1, self.atom_size)
        q_atoms = val + adv - adv.mean(dim=1, keepdim=True)
        dist = F.softmax(q_atoms, dim=-1)  # [B, A, atom_size]
        return dist
    
    def q_values(self, x):
        dist = self.forward(x)  # [B, A, atom_size]
        support = self.support.to(x.device)
        q = torch.sum(dist * support, dim=2)  # [B, A]
        return q
    
    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def projection_distribution(next_dist, rewards, dones, gamma, n, support, Vmin, Vmax, atom_size, device='cpu'):
        batch_size = rewards.size(0)
        delta_z = (Vmax - Vmin) / (atom_size - 1)
        # print("dones:", dones)
        # print("dones.dtype:", dones.dtype)
        # print("next_dist.shape:", next_dist.shape)
        support = support.unsqueeze(0).expand(batch_size, -1)  # [B, atom_size]
        Tz = rewards.unsqueeze(1) + gamma ** n * (1 - dones.unsqueeze(1)) * support  # [B, atom_size]
        Tz = Tz.clamp(Vmin, Vmax)
        b = (Tz - Vmin) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        l = l.clamp(0, atom_size - 1)
        u = u.clamp(0, atom_size - 1)

        # Distribute probabilities
        offset = torch.linspace(0, (batch_size - 1) * atom_size, batch_size).long().unsqueeze(1).to(device)  # [B, 1]
        l_idx = (l + offset).view(-1)
        u_idx = (u + offset).view(-1)

        m = next_dist  # [B, atom_size]
        proj_dist = torch.zeros_like(m)

        proj_dist_flat = proj_dist.view(-1)

        proj_dist_flat.index_add_(0, l_idx, (m * (u.float() - b)).view(-1))
        proj_dist_flat.index_add_(0, u_idx, (m * (b - l.float())).view(-1))        
        return proj_dist
    
    def compute_c51_loss(dist, target_dist, actions, weights=None):
        # dist: [B, A, atom_size]
        # actions: [B] → [B, 1, 1]
        actions = actions.view(-1, 1, 1).expand(-1, 1, dist.size(2))  # [B, 1, atom_size]
        chosen_action_dist = dist.gather(1, actions).squeeze(1)      # [B, atom_size]

        log_dist = torch.log(chosen_action_dist + 1e-8)
        loss = -torch.sum(target_dist * log_dist, dim=1)  # [B]

        if weights is not None:
            loss = loss * weights

        return loss.mean()

class DQNAgent:
    def __init__(self, state_dim, action_dim, buffer_capacity=200000, gamma=0.99, n=3, lr=3e-4, batch_size=64, hidden_dim=128, alpha=0.6, beta=0.4, sigma_init= 0.017, atom_size=101, Vmin=-80, Vmax=1800, log_dir="runs/your_run_name"):
        self.q_network = CategoricalDuelingNoisyNStepQNetwork(state_dim, action_dim, sigma_init=sigma_init, hidden_dim=hidden_dim, atom_size=atom_size, Vmin=Vmin, Vmax=Vmax)
        self.target_network = CategoricalDuelingNoisyNStepQNetwork(state_dim, action_dim, sigma_init=sigma_init, hidden_dim=hidden_dim, atom_size=atom_size, Vmin=Vmin, Vmax=Vmax)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.atom_size = self.q_network.atom_size
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = SumTreePrioritizedReplayBuffer(buffer_capacity, alpha=alpha)
        self.gamma = gamma
        self.n = n
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = beta  # IS重みの補正パラメータ
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.loss = 0.0
        self.td_error = 0.0

    def select_action(self, state):
        self.q_network.train()  # 明示的にtrainモードにしておく
        state = state.unsqueeze(0).to(self.device)  # [state_dim]
        # state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # [1, state_dim]
        with torch.no_grad():
            q_values = self.q_network.q_values(state)  # [1, action_dim]
            action = q_values.argmax(1).item()  # index of max Q
        return action, q_values.detach()
    
    def reset_noise(self):
        # ノイズリセット（NoisyNet用）
        self.q_network.reset_noise()
        self.target_network.reset_noise()  

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # サンプリング
        batch = self.replay_buffer.sample(self.batch_size, beta=self.beta)
        if batch is None:
            return  # データ不足時は学習をスキップ
        states, actions, rewards, next_states, dones, indices, weights = batch
        # テンソル化
        # states = torch.FloatTensor(states).to(self.device)
        # actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        # rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        # next_states = torch.FloatTensor(next_states).to(self.device)
        # dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        # weights = torch.FloatTensor(weights).to(self.device)
        states = states.to(self.device)  # [B, state_dim]
        actions = actions.unsqueeze(1).to(self.device)  # [B, 1]
        rewards = rewards.unsqueeze(1).to(self.device)  # [B, 1]
        next_states = next_states.to(self.device)  # [B, state_dim]
        dones = dones.unsqueeze(1).to(self.device)  # [B, 1]
        weights = weights.to(self.device)


        # target Q値の計算（Double DQN）
        with torch.no_grad():

            # 次状態のQ値の期待値から最良の行動を選ぶ
            next_q_values = self.q_network.q_values(next_states)  # [B, A]
            next_actions = next_q_values.argmax(1)  # [B]

            # 次状態の分布を取得し、対応する行動分だけ選ぶ
            next_dist = self.target_network.forward(next_states)  # [B, A, atom_size]
            next_actions = next_actions.unsqueeze(1).unsqueeze(1).expand(-1, 1, self.atom_size)  # [B, 1, atom_size]
            next_dist = next_dist.gather(1, next_actions).squeeze(1)  # [B, atom_size]

            # 分布の投影
            target_dist = CategoricalDuelingNoisyNStepQNetwork.projection_distribution(
                next_dist, rewards, dones, self.gamma, self.n, 
                self.q_network.support.to(self.device), 
                self.q_network.Vmin, self.q_network.Vmax, self.q_network.atom_size,
                device=self.device
            )

        # 現在の出力分布
        dist = self.q_network.forward(states)  # [B, A, atom_size]

        # ロス計算（クロスエントロピー）
        self.loss = CategoricalDuelingNoisyNStepQNetwork.compute_c51_loss(dist, target_dist, actions, weights)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        # if step  == 0:
        #     self.write_log("target_dist_argmax/Episode", target_dist[0].argmax(), epsode)

        # 優先度の更新
        with torch.no_grad():
            actions_exp = actions.view(-1, 1, 1).expand(-1, 1, dist.size(2))  # [64, 1, 51]
            chosen_dist = dist.gather(1, actions_exp).squeeze(1)  # [64, 51]            
            td_errors = torch.sum(torch.abs(chosen_dist.detach() - target_dist), dim=1).cpu().numpy()   
            self.td_error = np.mean(td_errors)  # 平均TD誤差を保存
        self.replay_buffer.update_priorities(indices, td_errors)

        # betaの更新（IS重みの補正パラメータ）        
        self.beta = min(1.0, self.beta + 0.000001)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def write_log(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
    
    def save_model(self, path="models/model", current_episode=0):  
        full_path = path + ".pth"
        # フォルダ部分を抽出して作成（存在しなければ）
        os.makedirs(os.path.dirname(full_path), exist_ok=True)        
        # モデル保存
        torch.save({'q_network_state_dict': self.q_network.state_dict(),
                    'episode': current_episode, 'beta': self.beta}, full_path)
        print(f"Models saved to {path}, episode: {current_episode}, beta: {self.beta}")

    def load_model(self, path="models/model"):
        full_path = path + ".pth"
        if os.path.exists(full_path):            
            checkpoint = torch.load(full_path)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])            
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.beta = checkpoint['beta']
            print(f"Model loaded from {full_path}")
            return checkpoint['episode']
        else:
            print(f"Model file not found at {full_path}, skipping load.")
            return 0

    def close(self):
        self.writer.close()
        print("TensorBoard writer closed.")

if( __name__ == "__main__"):    
    # 環境初期化
    # SEED = 51
    # random.seed(SEED)
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ChemicalEnvOnehot()
    sample_obs, _ = env.reset()
    flattened_obs = ChemicalEnvOnehot.flatten_observation(sample_obs)
    state_dim = flattened_obs.shape[0]
    action_dim = env.action_space.n
    n = 3  # nステップの数
    gamma = 0.99  # 割引率
    n_step_buffer = NStepBuffer(n=n, gamma=gamma)
    max_steps = 10000  # 1エピソードあたりの最大ステップ数
    # min_reward = -10  # 1エピソードあたりの最小報酬
    max_total_steps = 10000000  # 総ステップ数の上限
    save_step = max_total_steps // 10
    # DQNエージェントの初期化
    total_steps = 0
    best_reward = 0
    buffer_capacity = 100000
    learning_starts = 20000
    target_update_interval = 2000
    train_freq = 4
    alpha = 0.6
    beta = 0.4
    sigma_init = 0.17  # NoisyNetの初期標準偏差
    atom_size = 51
    Vmin = -1000
    Vmax = 6000
    hidden_dim = 256  # 隠れ層の次元数
    folder_name = f"models_alpha{(int)(alpha * 100)}_beta{(int)(beta * 100)}_sigma{(int)(sigma_init * 100)}_atoms{atom_size}_hidden{hidden_dim}_hidden{hidden_dim}_onehot_V2"
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, buffer_capacity=buffer_capacity, gamma=gamma, n=n, alpha=alpha, beta=beta, sigma_init=sigma_init, atom_size=atom_size, Vmin=Vmin, Vmax=Vmax, hidden_dim=hidden_dim, log_dir=f"runs/{folder_name}")
    episode = agent.load_model(f"{folder_name}/model")  # モデルのロード
    ema = 0.01
    ema_reward = 0.0
    ema_length = 0.0
    # ema_action_rate = 0.0
    ema_loss = 0.0
    ema_tderror = 0.0
    ema_step_reward = 0.0
    info = {}
    invalid_action_count = 0
    # メイン学習ループ
    for i in range(500000):
        obs, _ = env.reset()
        state = ChemicalEnvOnehot.flatten_observation(obs)
        done = False
        total_reward = 0
        step = 0        
        # actions = [0 for _ in range(action_dim)]    
        actions = np.zeros(action_dim, dtype=np.uint8)   
        x_field = np.zeros(6, dtype=np.uint8)
        q_values_var = []
        q_values_mean = []
        # rewards = [0 for _ in range(action_dim)]
        while not done:
            action, q_values = agent.select_action(state)   
            next_obs, reward, terminated, truncated, info = env.step(action)  
            x_field[info['x']] += 1
            x_field[info['x2']] += 1
            if info['invalid_action']:
                invalid_action_count += 1
            else:
                invalid_action_count = 0
            next_state = ChemicalEnvOnehot.flatten_observation(next_obs)
            done = terminated or truncated
            actions[action] += 1
            # rewards[action] += reward
            n_step_buffer.push(state, action, reward, next_state, done)

            if n_step_buffer.is_ready():
                n_state, n_action, n_reward, n_next_state, n_done = n_step_buffer.get()
                # エージェントのリプレイバッファに遷移を追加
                agent.replay_buffer.push(n_state, n_action, n_reward, n_next_state, n_done)
                n_step_buffer.pop()

            state = next_state
            total_reward += reward 
            total_steps += 1
            if total_steps > learning_starts:
                if total_steps % train_freq == 0:
                    agent.train_step()                    
                    ema_loss = ema * agent.loss.item() + (1 - ema) * ema_loss
                    ema_tderror = ema * agent.td_error + (1 - ema) * ema_tderror                    
                    agent.write_log("Loss/Step", ema_loss, total_steps)
                    agent.write_log("TDError/Step", ema_tderror, total_steps)
                if total_steps % target_update_interval == 0:
                    agent.update_target_network()
                if total_steps % save_step == 0:
                    agent.save_model(f"{folder_name}/model_step{total_steps}_ep{episode}", current_episode=episode) 
            agent.reset_noise()            
            # done = done or max_steps <= step or total_reward <= min_reward
            step += 1
            done = done or max_steps <= step or invalid_action_count >= n
            q_values_var.append(q_values.var().item())
            q_values_mean.append(q_values.mean().item())

        # 標本分散
        # 平坦化と計算
        q_values_var_np = np.array(q_values_var)
        q_values_mean_np = np.array(q_values_mean)
        q_values_mean_mean = q_values_mean_np.mean()
        q_values_var_total = np.mean(q_values_var_np**2 + (q_values_mean_np - q_values_mean_mean)**2)
        episode += 1         
        ema_reward = ema * total_reward + (1 - ema) * ema_reward 
        ema_length = ema * step + (1 - ema) * ema_length
        # max_action = max(actions)
        # ema_action_rate = ema * (max_action / step) + (1 - ema) * ema_action_rate
        # action_rate_2 = (actions[8] + actions[9]) / step * 100  # 8と9のアクションの合計頻度        
        # action_rate_1 = (actions[6] + actions[7] + actions[10] + actions[11]) / step * 100  # 6,7,10,11のアクションの合計頻度
        max_index = actions.argmax()  # 最大頻度のアクションのインデックス        
        max_action = actions[max_index] / step * 100  # 最大頻度のアクション数
        ema_step_reward = ema * (total_reward / step) + (1 - ema) * ema_step_reward
        print(f"{total_steps} Ep{episode}: Reward={total_reward}, Steps={step}, Most Action={max_index}:{max_action:.1f}%, Q Var={q_values_var_total:.4f}")
        # print(f"Episode {episode}: Total Reward = {total_reward}, Steps = {step}")
        print(info['field'].T[::-1])
        print(x_field)        
        if total_steps >= learning_starts:
            # 最高報酬記録
            if total_reward > best_reward:
                best_reward = total_reward
                agent.save_model(f"{folder_name}/model_best{best_reward}", current_episode=episode)

            # TensorBoardへのログ記録
            agent.write_log("Reward/Episode", ema_reward, episode)
            agent.write_log("Length/Episode", ema_length, episode)
            agent.write_log("StepReward/Episode", ema_step_reward, episode)
            # 各アクションの最大頻度をログに記録
            # agent.write_log("Actions/Episode", ema_action_rate, episode)

        if done:
            n_step_buffer.clear()
            if total_steps >= max_total_steps:
                print("Reached maximum total steps. Ending training.")
                break

    agent.save_model(f"{folder_name}/model", current_episode=episode)  # 最終モデル保存
    env.close()
    agent.close()
