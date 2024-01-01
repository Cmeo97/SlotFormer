import random
import sys
from typing import List, Optional, Union

from einops import rearrange
import numpy as np
import torch
from tqdm import tqdm
import wandb

from agent import Agent
from dataset import EpisodesDataset
from envs import SingleProcessEnv, MultiProcessEnv
from episode import Episode
from utils import EpisodeDirManager, RandomHeuristic


class Collector:
    def __init__(self, env: Union[SingleProcessEnv, MultiProcessEnv], dataset: EpisodesDataset, episode_dir_manager: EpisodeDirManager) -> None:
        self.env = env
        self.dataset = dataset
        self.episode_dir_manager = episode_dir_manager
        self.obs = self.env.reset()
        self.episode_ids = [None] * self.env.num_envs
        self.heuristic = RandomHeuristic(self.env.num_actions)

    @torch.no_grad()
    def collect(self, agent: Agent, epoch: int, epsilon: float, should_sample: bool, temperature: float, burn_in: int, *, num_steps: Optional[int] = None, num_episodes: Optional[int] = None):
        # assert self.env.num_actions == agent.world_model.act_vocab_size
        assert 0 <= epsilon <= 1

        assert (num_steps is None) != (num_episodes is None)
        should_stop = lambda steps, episodes: steps >= num_steps if num_steps is not None else episodes >= num_episodes

        to_log = []
        steps, episodes = 0, 0
        returns = []
        observations, actions, rewards, dones = [], [], [], []

        burnin_obs_rec, mask_padding = None, None
        if set(self.episode_ids) != {None} and burn_in > 0:
            current_episodes = [self.dataset.get_episode(episode_id) for episode_id in self.episode_ids]
            segmented_episodes = [episode.segment(start=len(episode) - burn_in, stop=len(episode), should_pad=True) for episode in current_episodes]
            mask_padding = torch.stack([episode.mask_padding for episode in segmented_episodes], dim=0).to(agent.device)
            burnin_obs = torch.stack([episode.observations for episode in segmented_episodes], dim=0).float().div(255).to(agent.device)
            burnin_obs_rec = torch.clamp(agent.tokenizer.encode_decode(burnin_obs, should_preprocess=True, should_postprocess=True), 0, 1)

        agent.actor_critic.reset(n=self.env.num_envs, burnin_observations=burnin_obs_rec, mask_padding=mask_padding)
        pbar = tqdm(total=num_steps if num_steps is not None else num_episodes, desc=f'Experience collection ({self.dataset.name})', file=sys.stdout)

        while not should_stop(steps, episodes):

            observations.append(self.obs)
            obs = rearrange(torch.FloatTensor(self.obs).div(255), 'n h w c -> n c h w').to(agent.device)
            act = agent.act(obs, should_sample=should_sample, temperature=temperature).cpu().numpy()

            if random.random() < epsilon:
                act = self.heuristic.act(obs).cpu().numpy()

            self.obs, reward, done, _ = self.env.step(act)

            # for billiard
            if "test" in self.dataset.name and steps >= 200:
                done = np.array([True] * self.env.num_envs)
                self.env.update_done_tracker(done)

            actions.append(act)
            rewards.append(reward)
            dones.append(done)

            new_steps = len(self.env.mask_new_dones)
            steps += new_steps
            pbar.update(new_steps if num_steps is not None else 0)

            # Warning: with EpisodicLifeEnv + MultiProcessEnv, reset is ignored if not a real done.
            # Thus, segments of experience following a life loss and preceding a general done are discarded.
            # Not a problem with a SingleProcessEnv.

            if self.env.should_reset():
                self.add_experience_to_dataset(observations, actions, rewards, dones)

                new_episodes = self.env.num_envs
                episodes += new_episodes
                pbar.update(new_episodes if num_episodes is not None else 0)

                for episode_id in self.episode_ids:
                    episode = self.dataset.get_episode(episode_id)
                    self.episode_dir_manager.save(episode, episode_id, epoch)
                    metrics_episode = {k: v for k, v in episode.compute_metrics().__dict__.items()}
                    metrics_episode['episode_num'] = episode_id
                    metrics_episode['action_histogram'] = wandb.Histogram(np_histogram=np.histogram(episode.actions.numpy(), bins=np.arange(0, self.env.num_actions + 1) - 0.5, density=True))
                    to_log.append({f'{self.dataset.name}/{k}': v for k, v in metrics_episode.items()})
                    returns.append(metrics_episode['episode_return'])

                self.obs = self.env.reset()
                self.episode_ids = [None] * self.env.num_envs
                agent.actor_critic.reset(n=self.env.num_envs)
                observations, actions, rewards, dones = [], [], [], []

        # Add incomplete episodes to dataset, and complete them later.
        if len(observations) > 0:
            self.add_experience_to_dataset(observations, actions, rewards, dones)

        agent.actor_critic.clear()

        metrics_collect = {
            '#episodes': len(self.dataset),
            '#steps': sum(map(len, self.dataset.episodes)),
        }
        if len(returns) > 0:
            metrics_collect['return'] = np.mean(returns)
        metrics_collect = {f'{self.dataset.name}/{k}': v for k, v in metrics_collect.items()}
        to_log.append(metrics_collect)

        return to_log

    def add_experience_to_dataset(self, observations: List[np.ndarray], actions: List[np.ndarray], rewards: List[np.ndarray], dones: List[np.ndarray]) -> None:
        assert len(observations) == len(actions) == len(rewards) == len(dones)
        for i, (o, a, r, d) in enumerate(zip(*map(lambda arr: np.swapaxes(arr, 0, 1), [observations, actions, rewards, dones]))):  # Make everything (N, T, ...) instead of (T, N, ...)
            episode = Episode(
                observations=torch.ByteTensor(o).permute(0, 3, 1, 2).contiguous(),  # channel-first
                actions=torch.LongTensor(a),
                rewards=torch.FloatTensor(r),
                ends=torch.LongTensor(d),
                mask_padding=torch.ones(d.shape[0], dtype=torch.bool),
            )
            if self.episode_ids[i] is None:
                self.episode_ids[i] = self.dataset.add_episode(episode)
            else:
                self.dataset.update_episode(self.episode_ids[i], episode)

import torch.nn.functional as F

class VIPERCollector(Collector):
    def __init__(self, env: Union[SingleProcessEnv, MultiProcessEnv], dataset: EpisodesDataset, episode_dir_manager: EpisodeDirManager) -> None:
        super().__init__(env, dataset, episode_dir_manager)

    def _prepare_tensors(self, obs: List[np.ndarray], actions: List[np.ndarray]) -> torch.FloatTensor:
        obs = torch.ByteTensor(np.array(obs)).div(255).permute(1, 0, 4, 2, 3).contiguous().float() # (1, T, C, H, W)
        actions = torch.LongTensor(np.array(actions)).permute(1, 0)  # (1, T)
        return obs, actions

    @torch.no_grad()
    def collect(self, agent: Agent, epoch: int, epsilon: float, should_sample: bool, temperature: float, burn_in: int, *, num_steps: Optional[int] = None, num_episodes: Optional[int] = None,
                context_length: Optional[int] = 5):
        # assert self.env.num_actions == agent.world_model.act_vocab_size
        assert 0 <= epsilon <= 1

        assert (num_steps is None) != (num_episodes is None)
        should_stop = lambda steps, episodes: steps >= num_steps if num_steps is not None else episodes >= num_episodes

        to_log = []
        steps, episodes = 0, 0
        returns = []
        observations, actions, rewards, dones = [], [], [], []

        burnin_obs_rec, mask_padding = None, None
        if set(self.episode_ids) != {None} and burn_in > 0:
            current_episodes = [self.dataset.get_episode(episode_id) for episode_id in self.episode_ids]
            segmented_episodes = [episode.segment(start=len(episode) - burn_in, stop=len(episode), should_pad=True) for episode in current_episodes]
            mask_padding = torch.stack([episode.mask_padding for episode in segmented_episodes], dim=0).to(agent.device)
            burnin_obs = torch.stack([episode.observations for episode in segmented_episodes], dim=0).float().div(255).to(agent.device)
            burnin_obs_rec = torch.clamp(agent.tokenizer.encode_decode(burnin_obs, should_preprocess=True, should_postprocess=True), 0, 1)

        agent.actor_critic.reset(n=self.env.num_envs, burnin_observations=burnin_obs_rec, mask_padding=mask_padding)
        pbar = tqdm(total=num_steps if num_steps is not None else num_episodes, desc=f'Experience collection ({self.dataset.name})', file=sys.stdout)

        while not should_stop(steps, episodes):

            observations.append(self.obs)
            obs = rearrange(torch.FloatTensor(self.obs).div(255), 'n h w c -> n c h w').to(agent.device)
            act = agent.act(obs, should_sample=should_sample, temperature=temperature).cpu().numpy()

            if random.random() < epsilon:
                act = self.heuristic.act(obs).cpu().numpy()

            cur_observations, cur_actions = self._prepare_tensors(observations, actions + [act])
            T = cur_observations.shape[1]
            cur_observations = cur_observations[:, max(0, T - context_length):]
            cur_actions = cur_actions[:, max(0, T - context_length):]
            mask_padding = torch.ones(cur_actions.shape[1], dtype=torch.bool)

            self.obs, reward, done, _ = self.env.step(act)

            #############################################
            # Fill in reward using log likelihood of the predition
            cur_observations, cur_actions, mask_padding = cur_observations.to(agent.device), cur_actions.to(agent.device), mask_padding.to(agent.device)
            obs_tokens = agent.tokenizer.encode(cur_observations, should_preprocess=True).tokens  # (BL, K)
            act_tokens = rearrange(cur_actions, 'b l -> b l 1')
            tokens = rearrange(torch.cat((obs_tokens, act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1))
            outputs = agent.world_model(tokens)

            mask_fill = torch.logical_not(mask_padding)
            labels_observations = rearrange(obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100), 'b t k -> b (t k)')[:, 1:].reshape(-1)
            logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
            loss_obs = F.cross_entropy(logits_observations, labels_observations)
            reward = loss_obs.item() + 0.25 * reward
            #############################################

            # for billiard
            if "test" in self.dataset.name and steps >= 200:
                done = np.array([True] * self.env.num_envs)
                self.env.update_done_tracker(done)

            actions.append(act)
            rewards.append(reward)
            dones.append(done)

            new_steps = len(self.env.mask_new_dones)
            steps += new_steps
            pbar.update(new_steps if num_steps is not None else 0)

            # Warning: with EpisodicLifeEnv + MultiProcessEnv, reset is ignored if not a real done.
            # Thus, segments of experience following a life loss and preceding a general done are discarded.
            # Not a problem with a SingleProcessEnv.

            if self.env.should_reset():
                self.add_experience_to_dataset(observations, actions, rewards, dones)

                new_episodes = self.env.num_envs
                episodes += new_episodes
                pbar.update(new_episodes if num_episodes is not None else 0)

                for episode_id in self.episode_ids:
                    episode = self.dataset.get_episode(episode_id)
                    self.episode_dir_manager.save(episode, episode_id, epoch)
                    metrics_episode = {k: v for k, v in episode.compute_metrics().__dict__.items()}
                    metrics_episode['episode_num'] = episode_id
                    metrics_episode['action_histogram'] = wandb.Histogram(np_histogram=np.histogram(episode.actions.numpy(), bins=np.arange(0, self.env.num_actions + 1) - 0.5, density=True))
                    to_log.append({f'{self.dataset.name}/{k}': v for k, v in metrics_episode.items()})
                    returns.append(metrics_episode['episode_return'])

                self.obs = self.env.reset()
                self.episode_ids = [None] * self.env.num_envs
                agent.actor_critic.reset(n=self.env.num_envs)
                observations, actions, rewards, dones = [], [], [], []

        # Add incomplete episodes to dataset, and complete them later.
        if len(observations) > 0:
            self.add_experience_to_dataset(observations, actions, rewards, dones)

        agent.actor_critic.clear()

        metrics_collect = {
            '#episodes': len(self.dataset),
            '#steps': sum(map(len, self.dataset.episodes)),
        }
        if len(returns) > 0:
            metrics_collect['return'] = np.mean(returns)
        metrics_collect = {f'{self.dataset.name}/{k}': v for k, v in metrics_collect.items()}
        to_log.append(metrics_collect)

        return to_log