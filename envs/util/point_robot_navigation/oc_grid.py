import torch as th
import numpy as np
from typing import Any, Dict, Optional, Type, Union, Tuple, List
from shapely.geometry import Polygon, MultiPolygon
import random
import shapely


def idx_to_position(idx: Union[List, np.ndarray], grid_length: float, position: str = "center") -> np.ndarray:
    '''
    Conversion from idx coordinate system to position
    Coordinate system is:
    [   (-1,1)      (1,1)
        (-1,-1)     (1,-1) ]
    obstacle_matrix indices:
    [   (0,0)       (0,1)
        (1,0)       (1,1)  ]
    @param idx: idx, has shape [batch_size, 2] -> 2 represents (row, column)
    @param grid_length: the length of a block in a grid
    @param position: Return either the center or the position of the left upper corner
    @return: position of the center of the idx referenced block. Has shape [batch_size, 2]
    '''
    if not isinstance(idx, np.ndarray):
        idx = np.array(idx).reshape(1, 2)
    elif len(idx.shape) == 1:
        idx = idx.reshape(1, 2)
    if idx.shape[1] > 2:  # idx of form [2, batch_size], convert to [batch_size, 2]
        idx = idx.T

    x_left_corner = grid_length * idx[:, 1] - 1  # x = h * column_idx - 1
    y_left_corner = -grid_length * idx[:, 0] + 1  # y = -h * row_idx + 1
    axis = 0 if len(x_left_corner.shape) == 1 else 1
    if position == "left_corner":
        return np.concatenate((x_left_corner, y_left_corner), axis=axis)
    # Center everything
    x_center = x_left_corner + grid_length / 2
    y_center = y_left_corner - grid_length / 2
    return np.concatenate((x_center, y_center), axis=axis)


def position_to_idx(x: Union[List, np.ndarray], grid_length: float, reference: str = "center") -> np.ndarray:
    """
    Position -1,1 is (0,0) in idx
    @param x: of shape [batch_size, 2]
    @param grid_length:
    @param reference:
    @return: idx of form [2, batch_size]: [0,:] is for rows, [1,:] is for columns
    """
    if not isinstance(x, np.ndarray):
        if len(x) == 1:
            print("hi")
        x = np.array(x).reshape(1, 2)
    elif len(x.shape) != 2:
        x = x.reshape(1, 2)

    if reference == "center":
        x_left_corner = x[:, 0] - grid_length / 2
        y_left_corner = x[:, 1] + grid_length / 2
    elif reference == "left_corner":
        x_left_corner, y_left_corner = x[:, 0], x[:, 1]
    else:
        raise NotImplementedError("Only <center> or <left_corner> is implemented. You chose: ", reference)

    column = np.around((x_left_corner + 1) / grid_length).astype(int)
    row = np.around((1 - y_left_corner) / grid_length).astype(int)
    # x is of shape [n, 2], n > 1
    if x.shape[0] > 1:
        column = column.reshape(1, -1)
        row = row.reshape(1, -1)
    return np.concatenate((row, column), axis=0)


def get_local_oc_grid(oc_grid: th.Tensor, oc_grid_view: int = 5) -> th.Tensor:
    """
    Function which extracts the local occupancy grid from a grid with marked position. Marks boundaries with 2 layers of
    1 and pads the rest with -1
    @param oc_grid: of shape: [batch_size, 2, n, n]. The dimension 2 consists of [0]: the oc grid, [1]: a gaussian blob
                    marking the position of the agent
    @param oc_grid_view: How many grids to left right, up and down to consider for local oc grid. F.e. if = 5, then
                        consider a 11x11 local grid
    @return: The local oc grid of shape [batch_size, 1, oc_grid_view * 2 + 1, oc_grid_view * 2 + 1].
    """

    batch_size = oc_grid.shape[0]

    idx = th.where(oc_grid[:, 1] > 0.99)
    idx = th.stack(idx, dim=0)[1:]
    # Mark boundary as 2 pads of 1
    x_padded = th.nn.functional.pad(oc_grid[:, 0], (2, 2, 2, 2), 'constant', 1)
    # Extend rest with -1
    pad = oc_grid_view - 2
    x_padded = th.nn.functional.pad(x_padded, (pad, pad, pad, pad), 'constant', -1)

    # idx marks left upper corner. Go 10 right and down for local grid
    shift = th.meshgrid(th.arange(oc_grid_view * 2 + 1), th.arange(oc_grid_view * 2 + 1))
    shift = th.stack(shift, dim=0).reshape(2, -1).to("cuda")
    n = shift.shape[1]

    # Mark Gaussian Blob in Real Grid
    batch_idx = th.cat([i * th.ones(n, dtype=int, device="cuda") for i in range(batch_size)]).unsqueeze(0)
    idx_combined = th.zeros((3, batch_size * n), dtype=int, device="cuda")
    idx_combined[0] = batch_idx
    for i in range(batch_size):
        idx_combined[1:, i * n: i * n + n] = idx[:, [i]] + shift
    x_local = x_padded[tuple(idx_combined)].reshape(batch_size, oc_grid_view * 2 + 1, oc_grid_view * 2 + 1)

    return x_local.unsqueeze(1)


def get_oc_grid_with_gaussian_position_blob(oc_grid, x, oc_grid_resolution):
    if len(x.shape) == 1:
        x = x.reshape(1, 2)
    batch_size = x.shape[0]
    oc_grid = np.expand_dims(oc_grid, axis=0) * 255  # 255 since img space is from 0-255
    oc_grid = np.repeat(oc_grid, batch_size, axis=0)  # dim = [batch_size, 40, 40]

    # Pad OC Grid, such that corner gaussian blobs can be put and then be cropped out
    padding = 2
    grid_gauss = np.zeros((batch_size, 40 + 2 * padding, 40 + 2 * padding))

    # Create Gaussian Blob centered around (0,0)
    x_gauss, y_gauss = np.meshgrid(np.linspace(-0.05, 0.05, 3), np.linspace(-0.05, 0.05, 3))
    dst = np.sqrt(x_gauss ** 2 + y_gauss ** 2)
    sigma = 0.05
    gauss = np.exp(-(dst ** 2 / (2.0 * sigma ** 2)))
    idx_middle = position_to_idx(x, 1 / oc_grid_resolution, reference="left_corner").reshape(2, -1)
    idx_middle = np.clip(idx_middle, a_min=0, a_max=oc_grid.shape[0] - 1)

    # Shift Gaussian Blob to right Position
    shift = np.mgrid[-1: 1.2: 1, -1: 1.2: 1].reshape(2, -1).astype(int)
    batch_idx = np.array([i * np.ones(shift.shape[1], dtype=int) for i in range(batch_size)]).reshape(1, -1)

    # Mark Gaussian Blob in Real Grid
    n = shift.shape[1]
    idx_combined = np.zeros((3, batch_size * n), dtype=int)
    idx_combined[0] = batch_idx
    for i in range(batch_size):
        idx_combined[1:, i * n: i * n + n] = idx_middle[:, i].reshape(2, 1) + shift + padding
    grid_gauss[tuple(idx_combined)] = np.tile(gauss[tuple(shift + 1)], batch_size)
    grid_gauss = grid_gauss[:, padding:-padding, padding:-padding] * 255  # Remove padding

    # Combine OC Grid with marked position
    grid = np.concatenate((np.expand_dims(oc_grid, 1), np.expand_dims(grid_gauss, 1)), axis=1)
    return grid


def get_local_oc_grid_numpy(oc_grid: np.ndarray, x: np.ndarray, grid_length: float = 0.05) -> np.ndarray:
    if len(x.shape) == 1:
        x = x.reshape(1, 2)
    batch_size = x.shape[0]

    # Bringing oc grid in right shape
    x_padded = np.pad(oc_grid, (5, 5), 'constant', constant_values=1)
    x_padded = np.expand_dims(x_padded, axis=0) * 255  # 255 since img space is from 0-255
    x_padded = np.repeat(x_padded, batch_size, axis=0)  # dim = [batch_size, 50, 50]

    # Finding corresponding index to position
    idx = position_to_idx(x, grid_length=grid_length, reference="left_corner")
    idx = np.clip(idx, a_min=0, a_max=1 / grid_length * 2 - 1)
    idx = idx.reshape(2, -1)

    # Selecting the right indices for the local oc grid
    shift = np.mgrid[0: 10.2: 1, 0: 10.2: 1].reshape(2, -1).astype(int)
    # idx marks left upper corner. Go 10 right and down for local grid
    batch_idx = np.array([i * np.ones(shift.shape[1], dtype=int) for i in range(batch_size)]).reshape(1, -1)
    # Mark Gaussian Blob in Real Grid
    n = shift.shape[1]
    idx_combined = np.zeros((3, batch_size * n), dtype=int)
    idx_combined[0] = batch_idx
    for i in range(batch_size):
        idx_combined[1:, i * n: i * n + n] = idx[:, [i]] + shift
    x_local = x_padded[tuple(idx_combined)].reshape(batch_size, 1, 11, 11)
    return x_local


def get_lidar_intersections(
        x: np.ndarray,
        obstacles: shapely.geometry.MultiPolygon,
        n_lidar_rays: int = 24,
        lidar_range: float = 0.5):
    """
    For one position x get all the lidar intersection points and distances
    @param x: Robot position
    @param obstacles: Shapely Polygons marking obstacles
    @param n_lidar_rays: Number of rays to spred out between 0 and 2pi
    @param lidar_range: Length of lidar rays
    @return: Intersection points of lidar and corresponding distances to robot a position x
    """
    # Preparation of n_lidar_rays spread out over 360 degree
    angles = np.arange(0, n_lidar_rays) / n_lidar_rays * 2 * np.pi
    end_points = x + lidar_range * np.concatenate(
        (np.cos(angles).reshape(-1, 1), np.sin(angles).reshape(-1, 1)), axis=1)
    intersection_points = np.empty((n_lidar_rays, 2))
    intersection_points[:] = np.nan

    # Looping over each lidar ray to get the closest intersection with an obstacle
    for i in range(n_lidar_rays):
        line = shapely.geometry.LineString([x, end_points[i]])
        intersection_line = obstacles.intersection(line)
        if not intersection_line.is_empty:
            if intersection_line.__class__.__name__ == 'LineString':
                intersection_points[i] = intersection_line.coords[0]
            elif intersection_line.__class__.__name__ == "Point":
                intersection_points[i] = intersection_line.coords[0]
            else:  # Multi String. Take first element which is closest
                intersection_points[i] = intersection_line[0].coords[0]

    # Calculating the object distance to the robot
    distances = -1 * np.ones(n_lidar_rays)
    idx = ~np.isnan(intersection_points[:, 0])
    distances[idx] = np.linalg.norm(x - intersection_points[idx], axis=1)

    return intersection_points, distances


def get_angle_x_axis(x):
    """
    Angle to x axis [1 0] calculated by arccos(x_axis dot x / |x|)
    Dot product of x * [1 0] selects the first element of x
    @param x: Vector of which the angle to x axis shall be calculated. Has shape [batch_size, 2]
    @return: The angle in radian
    """
    if len(x.shape) == 1:
        x = x.reshape(1, -1)

    angles = np.zeros(x.shape[0], dtype=np.float32)

    # If x is too close to origin, angle is set to zero
    idx_defined = np.where(np.linalg.norm(x, axis=1) > 0.01)
    x_defined = x[idx_defined]

    # Calculate angle
    angles[idx_defined] = np.arccos(x_defined[:, 0] / np.linalg.norm(x_defined, axis=1))
    # angles = np.arccos(x[:, 0] / np.linalg.norm(x, axis=1))
    # For y < 0 count angle in counterclockwise direction
    angles[x[:, 1] < 0] = 2 * np.pi - angles[x[:, 1] < 0]
    return angles


def get_length_trajectory(x: np.ndarray):
    """
    Getting the total length of a trajectory assuming linear interpolation between points
    :param x: Array of shape [n_points, [x,y]]
    :return: Length of trajectory
    """
    lengths = np.sqrt(np.sum(np.diff(x, axis=0) ** 2, axis=1))  # Length between corners
    total_length = np.sum(lengths)
    return total_length


# todo, not finished!
def get_lidar_intersection_parallel(x, obstacles, n_lidar_rays=24, lidar_range=0.5):
    angles = np.arange(0, n_lidar_rays) / n_lidar_rays * 2 * np.pi
    end_points = x + lidar_range * np.concatenate(
        (np.cos(angles).reshape(-1, 1), np.sin(angles).reshape(-1, 1)), axis=1)
    start_points = np.repeat(x.reshape(1, -1), n_lidar_rays, axis=0)

    # Bringing into right format for Multi line
    end_points = np.expand_dims(end_points, axis=1)
    start_points = np.expand_dims(start_points, axis=1)
    points = np.concatenate((start_points, end_points), axis=1)
    lines = shapely.geometry.MultiLineString(list(points))

    intersections = obstacles.intersection(lines)
    intersection_points = np.empty((len(intersections), 2))
    for i, intersection in enumerate(intersections):
        if intersection.__class__.__name__ == 'LineString':
            intersection_points[i] = intersection.coords[0]
        else:  # Multi String. Take first element which is closest
            intersection_points[i] = list(intersection[0].coords[0])

    # Todo find out why mapping does not work
    angle = get_angle_x_axis(intersection_points)
    idx = angle * n_lidar_rays / (2 * np.pi)
    # test_points = np.array([[1, 0], [1, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1]])
    # test_angles = get_angle_x_axis(test_points)
    # test_idx = test_angles * self.n_lidar_rays / (2 * np.pi)
    # test_angles = test_angles / np.pi * 180
    return intersection_points


def polygon_append(polygon: Union[None, Polygon], polygon_to_add: Polygon) -> Polygon:
    if polygon is None:
        polygon = polygon_to_add
    else:
        polygon = polygon.union(polygon_to_add)
    return polygon


def sample_position(low: float = 0.5, high: float = 1.0, rounding: float = 0.05) -> np.ndarray:
    """
    Sample state sucht that it is in a [-high,high] rectangle with a cutout  [-low, low] rectangle
    @param low: dimension of cut out rectangle
    @param high: dimension of rectangle to be cut out
    @param rounding: to which precision to round to
    @return:
    """

    x = (np.random.random(1) * 2 - 1) * high
    if abs(x) < low:  # Then y has to be in high, low
        y = np.random.random(1) * (high - low) + low
    else:  # No restriction on y
        y = (np.random.random(1) * 2 - 1) * high

    # Switch x and y such that there is no bias due to creation order
    state = random.choice([np.array([x, y]), np.array([y, x])]).squeeze()
    sign = 2 * np.random.randint(0, 2, 2) - 1
    state = sign * np.round(state / rounding) * rounding
    return state


def main():
    x = np.array([[1, 0],
                  [0, 1],
                  [-1, 0],
                  [0.0001, 0.001],
                  [0, -1],
                  [-0.001, 0.004]])
    angles = get_angle_x_axis(x)
    print("hi")


if __name__ == '__main__':
    main()
