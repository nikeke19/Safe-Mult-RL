import gym
from stable_baselines3.common.monitor import Monitor
from typing import Dict, Optional, Tuple, Union
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
import numpy as np
import time


class CarRacingMonitor(Monitor):
    """
    A custom monitor to also log collisions
    """

    def __init__(
            self,
            env: gym.Env,
            filename: Optional[str] = None,
            allow_early_resets: bool = True,
            reset_keywords: Tuple[str, ...] = (),
            info_keywords: Tuple[str, ...] = (),
            col_reward: float = -10,
    ):
        super(CarRacingMonitor, self).__init__(env, filename, allow_early_resets, reset_keywords, info_keywords)

        self.track_progress = []
        self.distances_before_crash = []
        self.velocities = []
        self.collision_preds = []
        self.col_reward = col_reward

        extra_keys = (
            'track_progress', 'velocity', 'var_velocity', 'soft_constraint_violation', 'boundary_distance',
            'var_boundary_distance', 'p_c', 'var_p_c', 'corr_p_c_distance', '1m_distance', 'p_c_1m_distance',
            'crash', 'velocities_before_end', 'p_c_before_end')

        self.results_writer.logger.fieldnames = self.results_writer.logger.fieldnames + extra_keys

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        ep_info_col = self.car_racing_step(info, done, reward)

        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            ep_info.update(ep_info_col)
            for key in self.info_keywords:
                ep_info[key] = info[key]

            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, done, info

    def car_racing_step(self, info, done, reward) -> Dict:
        self.track_progress.append(info['track_progress'])
        self.velocities.append(info['velocity'])
        self.distances_before_crash.append(info['distance_before_crash'])
        self.collision_preds.append(info['collision_pred'])

        if done:
            crash = False if reward > self.col_reward else True
            # Evaluate dangerous situations
            idx_small_distances = np.where(np.array(self.distances_before_crash) < 1)[0]
            n_dangerous_situation = len(idx_small_distances) / len(self.distances_before_crash)
            if n_dangerous_situation == 0:  # No dangerous situation
                col_probs_dangerous_situations = [0]
            else:
                col_probs_dangerous_situations = np.array(self.collision_preds)[idx_small_distances]

            velocities = np.array(self.velocities)

            return {'track_progress': info['track_progress'],
                    'velocity': velocities.mean(),
                    'var_velocity': velocities.var(),
                    'soft_constraint_violation': (velocities > 50).sum() / velocities.shape[0],
                    'boundary_distance': np.mean(self.distances_before_crash),
                    'var_boundary_distance': np.var(self.distances_before_crash),
                    'p_c': np.mean(self.collision_preds),
                    'var_p_c': np.var(self.collision_preds),
                    'corr_p_c_distance': np.corrcoef(self.collision_preds, self.distances_before_crash)[0, 1],
                    # Dangerous situations
                    '1m_distance': n_dangerous_situation,
                    'p_c_1m_distance': np.mean(col_probs_dangerous_situations),
                    # End of episode statistics
                    'crash': crash,
                    'velocities_before_end': np.mean(self.velocities[-15:]) if crash else 0,
                    'p_c_before_end': np.mean(self.collision_preds[-15:]) if crash else 0,
                    }
        return {}

    def reset(self, **kwargs) -> GymObs:
        self.track_progress = []
        self.distances_before_crash = []
        self.velocities = []
        self.collision_preds = []
        return super(CarRacingMonitor, self).reset(**kwargs)
