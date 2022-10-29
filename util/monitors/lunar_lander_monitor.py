import gym
from stable_baselines3.common.monitor import Monitor
from typing import Dict, List, Optional, Tuple, Union
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
import numpy as np
import time


class LunarLanderMonitor(Monitor):
    def __init__(
            self,
            env: gym.Env,
            filename: Optional[str] = None,
            allow_early_resets: bool = True,
            reset_keywords: Tuple[str, ...] = (),
            info_keywords: Tuple[str, ...] = (),
            col_reward: float = -100,
    ):
        super(LunarLanderMonitor, self).__init__(env, filename, allow_early_resets, reset_keywords, info_keywords)

        self.col_reward = col_reward

        self.landing_distances = []
        self.velocities_x = []
        self.velocities_y = []
        self.rolls = []
        self.d_rolls = []
        self.col_probs = []

        extra_keys = ('success', 'crash', 'timeout', 'mean_col_prob', 'var_col_prob',
                      # Stability and state properties
                      'final_landing_distance', 'mean_abs_vel_x', 'mean_vel_x', 'mean_vel_y', 'mean_abs_accel_x',
                      'mean_accel_x', 'mean_abs_accel_y', 'mean_accel_y', 'mean_roll', 'mean_d_roll', 'var_roll',
                      'var_d_roll', 'var_accel_x', 'var_accel_y',
                      # Statistics last 50 steps
                      'last25_landing_distance', 'last25_mean_abs_vel_x', 'last25_mean_vel_x', 'last25_mean_vel_y',
                      'last25_mean_abs_accel_x', 'last25_mean_accel_x', 'last25_mean_abs_accel_y',
                      'last25_mean_accel_y', 'last25_mean_roll', 'last25_mean_d_roll', 'last25_var_roll',
                      'last25_var_d_roll', 'last25_var_accel_x', 'last25_var_accel_y', 'last25_mean_col_prob',
                      'last25_var_col_prob',
                      # Correlations
                      'corr_vel_y_landing_distance', 'corr_vel_y_col_prob', 'corr_abs_roll_col_prob',
                      'corr_landing_distance_col_prob')

        self.results_writer.logger.fieldnames = self.results_writer.logger.fieldnames + extra_keys

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        ep_info_col = self.lunar_lander_step(info, done, reward)

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

    def lunar_lander_step_old(self, info, done, reward) -> Dict:
        self.velocities.append(info['velocity'])
        self.distances_before_crash.append(info['distance_before_crash'])
        self.angles.append(info['angle'])
        self.collision_preds.append(info['collision_pred'])

        if done:
            crash = False if reward > self.col_reward else True
            timeout = True if len(self.velocities) > 1000 else False
            # Evaluate dangerous situations
            idx_small_distances = np.where(np.array(self.distances_before_crash) < 1.5)[0]
            n_dangerous_situation = len(idx_small_distances)
            if n_dangerous_situation == 0:  # No dangerous situation
                col_probs_dangerous_situations = [0]
            else:
                col_probs_dangerous_situations = np.array(self.collision_preds)[idx_small_distances]

            return {'landing_distance': info['landing_distance'],
                    'avg_down_velocity': np.mean(self.velocities),
                    'avg_distance_before_crash': np.mean(self.distances_before_crash),
                    'avg_col_prob': np.mean(self.collision_preds),
                    'var_col_prob': np.var(self.collision_preds),
                    'var_angle': np.var(self.angles),
                    'var_down_velocity': np.var(self.velocities),
                    'correlation_col_distance': np.corrcoef(self.collision_preds, self.distances_before_crash)[0, 1],
                    # Dangerous situations
                    'n_dangerous_situations': n_dangerous_situation,
                    'col_probs_dangerous_situations': np.mean(col_probs_dangerous_situations),
                    # Crash situations or timeout
                    'crash': True if timeout else crash,
                    'timeout': timeout,
                    'vels_before_end': np.mean(self.velocities[-15:]) if crash else 0,
                    'col_prob_before_end': np.mean(self.collision_preds[-15:]) if crash else 0,
                    }

        return {}

    def lunar_lander_step(self, info, done, reward) -> Dict:
        self.landing_distances.append(info["landing_distance"])
        self.velocities_x.append(info['velocity_x'])
        self.velocities_y.append(info['velocity_y'])
        self.rolls.append(info["roll"])
        self.d_rolls.append(info["d_roll"])
        self.col_probs.append(info['col_prob'])

        if done:
            crash = False if reward > self.col_reward else True
            timeout = True if len(self.velocities_x) > 1000 else False
            a_x = np.array(self.velocities_x)[1:] - np.array(self.velocities_x)[:-1]
            a_y = np.array(self.velocities_y)[1:] - np.array(self.velocities_y)[:-1]

            idx_last25 = -len(self.rolls) // 4
            return {
                # Key statistics
                'success': 1 - (crash + timeout),
                'crash': crash,
                'timeout': timeout,
                'mean_col_prob': np.mean(self.col_probs),
                'var_col_prob': np.var(self.col_probs),
                # Stability and state properties
                'final_landing_distance': info['landing_distance'],
                'mean_abs_vel_x': np.mean(np.abs(self.velocities_x)),
                'mean_vel_x': np.mean(self.velocities_x),
                'mean_vel_y': np.mean(self.velocities_y),
                'mean_abs_accel_x': np.mean(np.abs(a_x)),
                'mean_accel_x': np.mean(a_x),
                'mean_abs_accel_y': np.mean(np.abs(a_y)),
                'mean_accel_y': np.mean(np.abs(a_y)),
                'mean_roll': np.mean(self.rolls),
                'mean_d_roll': np.mean(self.d_rolls),
                'var_roll': np.var(self.rolls),
                'var_d_roll': np.var(self.d_rolls),
                'var_accel_x': np.var(a_x),
                'var_accel_y': np.var(a_y),
                # Statistics last 50 steps
                'last25_landing_distance': self.landing_distances[-50],
                'last25_mean_abs_vel_x': np.mean(np.abs(self.velocities_x[idx_last25:])),
                'last25_mean_vel_x': np.mean(self.velocities_x[idx_last25:]),
                'last25_mean_vel_y': np.mean(self.velocities_y[idx_last25:]),
                'last25_mean_abs_accel_x': np.mean(np.abs(a_x[idx_last25:])),
                'last25_mean_accel_x': np.mean(a_x[idx_last25:]),
                'last25_mean_abs_accel_y': np.mean(np.abs(a_y[idx_last25:])),
                'last25_mean_accel_y': np.mean(np.abs(a_y[idx_last25:])),
                'last25_mean_roll': np.mean(self.rolls[idx_last25:]),
                'last25_mean_d_roll': np.mean(self.d_rolls[idx_last25:]),
                'last25_var_roll': np.var(self.rolls[idx_last25:]),
                'last25_var_d_roll': np.var(self.d_rolls[idx_last25:]),
                'last25_var_accel_x': np.var(a_x[idx_last25:]),
                'last25_var_accel_y': np.var(a_y[idx_last25:]),
                'last25_mean_col_prob': np.mean(self.col_probs[idx_last25:]),
                'last25_var_col_prob': np.var(self.col_probs[idx_last25:]),
                # Correlations
                'corr_vel_y_landing_distance': np.corrcoef(self.velocities_y, self.landing_distances)[0, 1],
                'corr_vel_y_col_prob': np.corrcoef(self.velocities_y, self.col_probs)[0, 1],
                'corr_abs_roll_col_prob': np.corrcoef(np.abs(self.rolls), self.col_probs)[0, 1],
                'corr_landing_distance_col_prob': np.corrcoef(self.landing_distances, self.col_probs)[0, 1],
            }

        return {}

    def reset(self, **kwargs) -> GymObs:
        # self.angles = []
        # self.distances_before_crash = []
        # self.velocities = []
        # self.collision_preds = []

        self.landing_distances = []
        self.velocities_x = []
        self.velocities_y = []
        self.rolls = []
        self.d_rolls = []
        self.col_probs = []

        return super(LunarLanderMonitor, self).reset()
