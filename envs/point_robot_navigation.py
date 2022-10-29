import time

from envs.util.point_robot_navigation.base_box_env import BaseBoxEnv
import numpy as np
import shapely.geometry
import random
from typing import Any, Dict, Optional, Type, Union, Tuple, List
import matplotlib.pyplot as plt
import descartes
import matplotlib.patches as patches
from gym.spaces import Box
from gym import spaces
from shapely.geometry import Polygon
from envs.util.point_robot_navigation.oc_grid import (
    get_local_oc_grid_numpy,
    polygon_append,
    position_to_idx,
    get_lidar_intersections,
    get_lidar_intersection_parallel)


class RandomBoxEnv(BaseBoxEnv):
    """
    Box Random with local occupancy grid and either oc grid or lidar observation space
    """

    def __init__(self,
                 goal: np.ndarray = None,
                 sparse_reward: bool = True,
                 observation_space: str = "oc_grid",
                 step_penalty: float = 0.1,
                 n_lidar_rays: int = 24,
                 verbose: bool = False):
        super(RandomBoxEnv, self).__init__(None, goal, sparse_reward)

        self.oc_grid_resolution = 1 / 0.05
        self.observation_space_type = observation_space
        self.n_lidar_rays = n_lidar_rays

        self.verbose = verbose
        self.obstacles = []
        self.polygons_obstacle_construction = None
        self.polygons_collision_check = None
        self.polygons_lidar_check = None
        self.state = None
        self.oc_grid = None
        self.step_penalty = step_penalty

        # Setting up Observation space
        self.state_space = Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        if self.observation_space_type == "oc_grid":
            self.image_space = Box(low=0, high=255, shape=(1, 11, 11), dtype=np.uint8)
            self.observation_space = spaces.Dict(spaces={"img": self.image_space, "vec": self.state_space})
        else:  # Lidar Space
            ones = np.ones(n_lidar_rays + 2)
            self.observation_space = Box(low=-ones, high=ones, dtype=np.float32)

        # Setup polygons to check for intersection
        s = 0.1
        self.goal_polygon = Polygon([[-s, s], [s, s], [s, -s], [-s, -s]])

    def seed_spaces(self, seed):
        self.state_space.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def reset(self, mode="train", start_state=None, obstacles=None):

        # Reset variables
        self.obstacles = []
        self.polygons_obstacle_construction, self.polygons_collision_check = None, None

        # Construct obstacles
        if obstacles is None:
            self.construct_random_obstacles()
        else:
            self._get_obstacles_from_list(obstacles)
        self.create_occupancy_grid()

        # Define Starting State
        self.set_starting_position(start_state)
        self.state = self.get_image_state(self.current_position)
        if self.observation_space_type == "oc_grid":
            self.state = {"img": self.state, "vec": self.current_position}
        else:
            self.state = np.concatenate((self.current_position, self.state), axis=0)  # todo check right dim
        self.trajectory = self.current_position.reshape(1, -1)
        return self.state

    def step(self, actions):
        next_state, reward, done, info = super(RandomBoxEnv, self).step(actions)

        if reward >= 0.99 * self.final_reward:
            reward *= 2

        # Add a little step cost
        elif not self.sparse_reward:
            reward -= self.step_penalty

        self.state = self.get_image_state(self.current_position)
        if self.observation_space_type == "oc_grid":
            self.state = {"img": self.state, "vec": self.current_position}
        else:
            self.state = np.concatenate((self.current_position, self.state), axis=0)  # todo check right dim
        return self.state, reward, done, info

    def is_collision(self, state, next_state):
        p_2 = shapely.geometry.Point(next_state[0], next_state[1])
        if (self.polygons_collision_check.contains(p_2)
                or shapely.geometry.LineString([state, next_state]).intersects(self.polygons_collision_check)):
            return True
        elif (np.abs(next_state[0:2]) > 1).any():  # Leaving the grid
            return True
        else:  # No collision
            return False

    def minimum_distance_to_obstacle(self, point):
        point_shapely = shapely.geometry.Point(point)
        distance_from_boundary = min(1 - np.abs(point))
        distances = [max(0, distance_from_boundary)]
        for obstacle in self.obstacles:
            polygon = obstacle["polygon"]
            distance_from_obstacle = polygon.exterior.distance(point_shapely)
            distances.append(distance_from_obstacle)
        return min(distances)

    def set_starting_position(self, starting_state: np.ndarray = None):
        if starting_state is None:
            starting_state = self.state_space.sample()
            while self.polygons_obstacle_construction.contains(shapely.geometry.Point(starting_state)):
                starting_state = self.state_space.sample()

        self.current_position = starting_state
        self.last_distance2goal = np.linalg.norm(self.current_position - self.goal)

    def construct_random_obstacles(self):
        n_blocks = np.random.randint(2, 5)
        n_walls = np.random.randint(4, 6)

        wall_count = 0
        for i in range(100):
            l = random.choice([0.6, 0.5, 0.4])
            orientation = random.choice(["vertical", "horizontal"])
            height = l if orientation == "vertical" else 0.1
            width = l if orientation == "horizontal" else 0.1

            # Random sample center rounded to 1/self.oc_grid_resolution -> round to 0.05
            center = np.random.random(2) * 1.4 - 0.7  # In [-0.7,0.7]
            x, y = np.round(center * self.oc_grid_resolution) / self.oc_grid_resolution

            # Putting points clock wise starting from upper left corner
            h, w = height / 2, width / 2
            points = [[x - w, y + h], [x + w, y + h], [x + w, y - h], [x - w, y - h]]
            obstacle = {"height": height, "width": width, "center": np.array([x, y]), "polygon": Polygon(points)}

            if self.is_obstacle_feasible(obstacle, safety_distance=0.15):
                wall_count += 1
                self.obstacles.append(obstacle)
                self.polygons_collision_check = polygon_append(self.polygons_collision_check, obstacle["polygon"])
                if wall_count == n_walls:
                    break

            if i == 99 and self.verbose:
                print("Reached max iterations, but could not create all walls")

        block_count = 0
        for i in range(100):
            l = random.choice([0.1, 0.2, 0.3])

            # Random sample center rounded to 1/self.oc_grid_resolution -> round to 0.05
            center = np.random.random(2) * 1.6 - 0.8  # In [-0.7,0.7]
            x, y = np.round(center * self.oc_grid_resolution) / self.oc_grid_resolution

            # Putting points clock wise starting from upper left corner
            h = w = l / 2
            points = [[x - w, y + h], [x + w, y + h], [x + w, y - h], [x - w, y - h]]
            obstacle = {"height": l, "width": l, "center": np.array([x, y]), "polygon": Polygon(points)}

            if self.is_obstacle_feasible(obstacle):
                block_count += 1
                self.obstacles.append(obstacle)
                self.polygons_collision_check = polygon_append(self.polygons_collision_check, obstacle["polygon"])
                if block_count == n_blocks:
                    break

            if i == 99 and self.verbose:
                print("Reached max iterations, but could not create all blocks")

        # if self.observation_space_type == "lidar":
        # Adding borders of environment as polygons
        polygon_left = Polygon([[1.2, 1.2], [1.2, 1], [-1, 1], [-1, -1.2], [-1.2, -1.2], [-1.2, 1.2]])
        points_right = Polygon([[1, 1.2], [1.2, 1.2], [1.2, -1.2], [-1.2, -1.2], [-1.2, -1], [1, -1]])
        polygon_boundary = polygon_append(polygon_left, points_right)
        self.polygons_lidar_check = polygon_append(self.polygons_collision_check, polygon_boundary)

    def is_obstacle_feasible(self, obstacle: dict, safety_distance: float = 0.1) -> bool:
        # Create a polygon which is slightly larger
        x, y = obstacle["center"]
        h = obstacle["height"] / 2 + safety_distance
        w = obstacle["width"] / 2 + safety_distance
        points = [[x - w, y + h], [x + w, y + h], [x + w, y - h], [x - w, y - h]]
        points = np.array(points)
        polygon = Polygon(points)

        # Check for out of boundary:
        if (np.abs(points) > 1).any():
            return False

        # Check for intersection with goal
        if self.goal_polygon.intersects(polygon):
            return False

        # Check for intersection with other obstacle or goal
        if self.polygons_obstacle_construction is not None and self.polygons_obstacle_construction.intersects(polygon):
            return False

        # If come to here, obstacle is feasible: Add to list of polygons to check
        self.polygons_obstacle_construction = polygon_append(self.polygons_obstacle_construction, polygon)
        return True

    def create_occupancy_grid(self):
        n = int(2 * self.oc_grid_resolution)
        self.oc_grid = np.zeros((n, n), dtype=int)
        for obstacle in self.obstacles:
            x, y = obstacle["center"]
            h, w = obstacle["height"], obstacle["width"]
            # Finding the left upper corner of the obstacle to know where to start
            left_corner = [x - w / 2, y + h / 2]
            start_row, start_column = position_to_idx(left_corner, 1 / self.oc_grid_resolution, reference="left_corner")
            # How far to go
            end_row = start_row + int(h * self.oc_grid_resolution)
            end_column = start_column + int(w * self.oc_grid_resolution)
            # Marking as Occupied
            self.oc_grid[start_row: end_row, start_column: end_column] = 1

    def get_image_state(self, x):
        if self.observation_space_type == "oc_grid":
            return get_local_oc_grid_numpy(self.oc_grid, x, grid_length=1 / self.oc_grid_resolution)
        else:
            return self.get_lidar_state(x)

    def get_lidar_state(self, x):
        if len(x.shape) == 1:  # Batch size 1
            return get_lidar_intersections(x, self.polygons_lidar_check, self.n_lidar_rays, lidar_range=0.5)[1]
        else:  # Multi Batch
            states = np.empty((x.shape[0], self.n_lidar_rays))
            for i in range(x.shape[0]):
                states[i] = \
                    get_lidar_intersections(x[i], self.polygons_lidar_check, self.n_lidar_rays, lidar_range=0.5)[1]
            return states

    def _get_obstacles_from_list(self, obstacles):
        self.obstacles = []
        self.polygons_collision_check, self.polygons_obstacle_construction = None, None
        # Prepare obstacles:
        for i, obstacle in enumerate(obstacles):
            h, w = obstacle["height"] / 2, obstacle["width"] / 2
            x, y = obstacle["center"]
            points = [[x - w, y + h], [x + w, y + h], [x + w, y - h], [x - w, y - h]]
            obstacles[i]["polygon"] = Polygon(points)
            self.obstacles.append(obstacles[i])
            self.polygons_collision_check = polygon_append(self.polygons_collision_check, obstacles[i]["polygon"])
        self.create_occupancy_grid()

        # Adding borders of environment as polygons for lidar
        polygon_left = Polygon([[1.2, 1.2], [1.2, 1], [-1, 1], [-1, -1.2], [-1.2, -1.2], [-1.2, 1.2]])
        points_right = Polygon([[1, 1.2], [1.2, 1.2], [1.2, -1.2], [-1.2, -1.2], [-1.2, -1], [1, -1]])
        polygon_boundary = polygon_append(polygon_left, points_right)
        self.polygons_lidar_check = polygon_append(self.polygons_collision_check, polygon_boundary)

        # Create polygons_collision_check
        safety_distance = 0.15
        for i, obstacle in enumerate(obstacles):
            x, y = obstacle["center"]
            h = obstacle["height"] / 2 + safety_distance
            w = obstacle["width"] / 2 + safety_distance
            points = [[x - w, y + h], [x + w, y + h], [x + w, y - h], [x - w, y - h]]
            points = np.array(points)
            polygon = Polygon(points)
            self.polygons_obstacle_construction = polygon_append(self.polygons_obstacle_construction, polygon)

    def render(self, mode="human"):
        plt.style.use("classic")
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.add_patch(plt.Circle(tuple(self.goal), self.eps, facecolor='none', linewidth=1, alpha=1, edgecolor="black"))
        ax.add_patch(
            descartes.PolygonPatch(self.polygons_collision_check, fc='black', alpha=0.5, fill=True, edgecolor="red"))

        # plt.xticks(ticks=np.arange(self.x_min, self.x_max + 1), labels=np.arange(self.x_min, self.x_max + 1))
        # plt.yticks(ticks=np.arange(self.y_min, self.y_max + 1), labels=np.arange(self.y_min, self.y_max + 1))
        ax.set_aspect('equal', 'box')
        ax.set(xlim=(-1, 1), ylim=(-1, 1))

        if self.observation_space_type == "oc_grid":
            ax.imshow(self.oc_grid, extent=[-1, 1, -1, 1], alpha=0.5)
        ax.scatter(self.trajectory[0, 0], self.trajectory[0, 1], c="green", s=20)
        plt.plot(self.trajectory[:, 0], self.trajectory[:, 1], linewidth=0.5, color="blue")
        plt.show()

    def debug_collision_check(self):
        self.reset()
        x_no_col = np.empty((0, 2), dtype=np.float32)
        x_col = np.empty((0, 2), dtype=np.float32)
        for i in range(10000):
            x = 2 * np.random.sample((2,)) - 1  # number between -1,1
            if self.is_collision(x, x):
                x_col = np.vstack((x_col, x))
            else:
                x_no_col = np.vstack((x_no_col, x))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.add_patch(plt.Circle(tuple(self.goal), self.eps, facecolor='none', linewidth=1, alpha=1, edgecolor="black"))
        ax.add_patch(
            descartes.PolygonPatch(self.polygons_collision_check, fc='black', alpha=0.5, fill=True, edgecolor="red"))
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.imshow(self.oc_grid, extent=[-1, 1, -1, 1], alpha=0.5)
        ax.scatter(x_col[:, 0], x_col[:, 1], c="red", alpha=1, s=1)
        ax.scatter(x_no_col[:, 0], x_no_col[:, 1], c="green", alpha=1, s=1)
        plt.show()

    def debug_lidar(self):
        lidar_range = 0.5
        self.reset()

        start = time.time_ns()
        intersection_points, distances = get_lidar_intersections(self.current_position, self.polygons_lidar_check)
        print("needed time:", time.time_ns() - start)
        start = time.time_ns()
        test = get_lidar_intersection_parallel(self.current_position, self.polygons_lidar_check)
        print("needed time:", time.time_ns() - start)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(self.trajectory[0, 0], self.trajectory[0, 1], c="green", s=20)
        ax.add_patch(
            descartes.PolygonPatch(self.polygons_collision_check, fc='black', alpha=0.5, fill=True, edgecolor="red"))

        angles = np.arange(0, self.n_lidar_rays) / self.n_lidar_rays * 2 * np.pi
        end_points = self.current_position + lidar_range * np.concatenate(
            (np.cos(angles).reshape(-1, 1), np.sin(angles).reshape(-1, 1)), axis=1)
        start_points = np.repeat(self.current_position.reshape(1, -1), self.n_lidar_rays, axis=0)

        x = [start_points[:, 0], end_points[:, 0]]
        y = [start_points[:, 1], end_points[:, 1]]
        plt.plot(x, y, color="r", linewidth=0.2)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_title(len(test))

        obstacle_points = intersection_points[~np.isnan(intersection_points)].reshape(-1, 2)
        if obstacle_points.shape[0] > 0:
            ax.scatter(obstacle_points[:, 0], obstacle_points[:, 1], c="blue", s=10)
        plt.show()

    def debug_trajectory_plot(self):
        obstacles = [{"height": 0.3, "width": 0.3, "center": np.array([-0.65, 0.65])},
                     {"height": 0.5, "width": 0.1, "center": np.array([-0.15, 0.55])},
                     {"height": 0.1, "width": 0.4, "center": np.array([0.45, 0.6])},
                     {"height": 0.2, "width": 0.2, "center": np.array([-0.45, 0])},
                     {"height": 0.4, "width": 0.1, "center": np.array([0.45, -0.05])},
                     {"height": 0.1, "width": 0.1, "center": np.array([0, -0.3])},
                     {"height": 0.1, "width": 0.5, "center": np.array([-0.5, -0.7])},
                     {"height": 0.2, "width": 0.2, "center": np.array([0.5, -0.65])}]
        start_state = np.array([-0.5, -0.35])
        self.reset("train", start_state, obstacles)
        for i in range(5000):
            a = -1 + 2 * np.random.random_sample(2)
            state, reward, done, info = env.step(a)
            if done:
                break
        env.render()


if __name__ == '__main__':
    env = RandomBoxEnv(observation_space="oc_grid")
    env.debug_trajectory_plot()
    # env.debug_lidar()

    # env.debug_collision_check()
    # env.reset()
    # total_reward = 0
    # for i in range(5000):
    #     a = -1 + 2 * np.random.random_sample(2)
    #     state, reward, done, info = env.step(a)
    #     total_reward += reward
    #     if done:
    #         print("Done at iteration: ", i)
    #         break
    # print(total_reward)
    # env.render()
