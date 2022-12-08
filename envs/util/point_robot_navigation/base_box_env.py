import os
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shapely.geometry
import descartes
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class BaseBoxEnv(gym.Env):
    def __init__(self,
                 obstacles=None,
                 goal=None,
                 sparse_reward=False):

        # Set up environment
        if obstacles is None:
            obstacle_1 = [[-0.8, -0.2], [-0.4, -0.4], [-0.6, -0.4]]
            obstacle_2 = [[0.2, -0.2], [0.6, -0.4], [0.5, -0.7], [0.0, -0.4]]
            obstacle_3 = [[0, 0.8], [0.2, 0.6], [0.1, 0.5], [-0.1, 0.5], [-0.2, 0.6]]
            obstacles = [obstacle_1, obstacle_2, obstacle_3]
        self.resolution = 100
        self.x_min, self.x_max = -1, 1
        self.y_min, self.y_max = -1, 1
        self.obstacles = self.get_obstacle(obstacles)
        self.eps = 0.1
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([self.x_min, self.y_min]), high=np.array([self.x_max, self.y_max]), dtype=np.float32)
        self.goal = np.array([0, 0]) if goal is None else goal

        # Set up Dynamics
        self.A = np.eye(2)
        self.B = 0.05 * np.eye(2)
        self.last_distance2goal = None
        self.current_position = None

        # Set up reward structure
        self.sparse_reward = sparse_reward
        self.final_reward = 20
        self.stepcost = -self.final_reward / 200

        # Logging
        self.q_action, self.c_action = -1, -1
        self.q, self.c = [-1, -1, -1, -1], [-1, -1, -1, -1]
        self.heat_map = None
        self.resolution_heat_map = self.resolution
        self.trajectory = None

        # Init
        self.seed()
        self.init_visit_heat_map()

    def init_visit_heat_map(self):
        n = (self.observation_space.high[0] - self.observation_space.low[0]) * self.resolution_heat_map + 1
        n = round(n)
        self.heat_map = np.zeros((n, n))

    def update_heat_map(self, x):
        idx = (x + 1) * self.resolution_heat_map
        # Through rounding, value can be sometimes outside grid -> Clip
        idx = np.clip(idx, a_min=0, a_max=self.heat_map.shape[0] - 1)
        idx = idx.astype(int)
        self.heat_map[idx[0], idx[1]] += 1

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_obstacle(self, obstacle_list):
        polygon = shapely.geometry.Polygon(obstacle_list[0])
        for i in range(1, len(obstacle_list)):
            new_polygon = shapely.geometry.Polygon(obstacle_list[i])
            polygon = polygon.union(new_polygon)

        return polygon

    def render(self, mode="human"):
        plt.style.use("classic")
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.add_patch(descartes.PolygonPatch(self.obstacles, fc='black', alpha=0.5))
        ax.add_patch(patches.Circle((self.goal[0], self.goal[1]), self.eps, facecolor='none', linewidth=1, alpha=1))
        plt.xticks(ticks=np.arange(self.x_min, self.x_max + 1), labels=np.arange(self.x_min, self.x_max + 1))
        plt.yticks(ticks=np.arange(self.y_min, self.y_max + 1), labels=np.arange(self.y_min, self.y_max + 1))
        plt.imshow(np.ones((self.resolution + 1, self.resolution + 1)), "gray",
                   extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                   vmin=0, vmax=1)
        ax.scatter(self.goal[0], self.goal[1], s=5, color="red")
        plt.plot(self.trajectory[:, 0], self.trajectory[:, 1], linewidth=0.5, color="blue")
        plt.show()

    def reset(self, mode="train", start_state=None):
        starting_state = self.observation_space.sample()
        s_state = shapely.geometry.Point(starting_state)

        while self.obstacles.contains(s_state):
            starting_state = self.observation_space.sample()
            s_state = shapely.geometry.Point(starting_state)

        if start_state is not None:
            starting_state = start_state

        self.current_position = starting_state
        self.last_distance2goal = np.linalg.norm(self.current_position - self.goal)
        self.trajectory = starting_state.reshape(1, -1)
        return self.current_position

    def is_collision(self, state, next_state):
        p_2 = shapely.geometry.Point(next_state[0], next_state[1])
        if self.obstacles.contains(p_2) or shapely.geometry.LineString([state, next_state]).intersects(self.obstacles):
            return True
        # Leaving the grid
        elif (np.abs(next_state[0:2]) > 1).any():
            return True
        # No collision
        else:
            return False

    def goal_reached(self, state):
        return np.linalg.norm(state[:2] - self.goal[:2]) < self.eps

    def minimum_distance_to_obstacle(self, point):
        point_shapely = shapely.geometry.Point(point)
        distances = []
        for polygon in self.obstacles:
            distance_from_obstacle = polygon.exterior.distance(point_shapely)
            distance_from_boundary = min(1 - np.abs(point))
            distances.append(min(distance_from_obstacle, distance_from_boundary))
        return min(distances)

    def step(self, actions):
        actions = np.array(actions).reshape(-1)
        next_state = self.A @ self.current_position + self.B @ actions
        self.update_heat_map(next_state)  # Inform about visited state
        current_distance2goal = np.linalg.norm(next_state - self.goal)
        info = {'distance_before_crash': self.minimum_distance_to_obstacle(next_state),
                'collision_pred': self.c_action,
                'n_steps': len(self.trajectory),
                'distance_from_goal': current_distance2goal}

        done = False
        if self.sparse_reward:
            reward = - self.final_reward / 200
        else:
            reward = self.final_reward * (self.last_distance2goal - current_distance2goal)

        # Goal reached
        if self.goal_reached(self.current_position) or self.goal_reached(next_state):
            done = True
            reward = self.final_reward
        # Collision
        elif self.is_collision(self.current_position, next_state):
            done = True
            reward = -self.final_reward
        # Timeout
        elif len(self.trajectory) > 1000:
            done = True

        # Update
        self.last_distance2goal = current_distance2goal
        self.current_position = next_state
        self.trajectory = np.vstack((self.trajectory, next_state))
        return next_state, reward, done, info

    def update_critic_values(self, q_action, c_action, q, c):
        self.q_action, self.c_action, self.q, self.c = q_action, c_action, q, c


def main():
    env = BaseBoxEnv(goal=np.array([0, 0]))
    env.reset()
    total_reward = 0
    for i in range(5000):
        a = -1 + 2 * np.random.random_sample(2)
        state, reward, done, info = env.step(a)
        total_reward += reward
        if done:
            print("Done at iteration: ", i)
            break
    print(total_reward)
    env.render()


if __name__ == '__main__':
    main()
