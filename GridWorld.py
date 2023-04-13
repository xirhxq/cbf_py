import math
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import clip_by_rect
import matplotlib.pyplot as plt
import time


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'Function {func.__name__} run time: {end_time - start_time} s')
        return result

    return wrapper


def io_decorator(func):
    def wrapper(*args, **kwargs):
        print(f'Function {func.__name__} input: {args[1]}, {kwargs}')
        result = func(*args, **kwargs)
        print(f'Function {func.__name__} output: {result}')
        return result

    return wrapper


def is_point_in_polygon(d, p):
    dp = np.diff(p, axis=0)
    exdp = dp[:, np.newaxis, :]
    exd = d[np.newaxis, :, :]
    dminusp = exd - p[:-1, np.newaxis, :]
    cross = np.cross(exdp, dminusp)
    inside = np.all(cross > 0, axis=0) | np.all(cross < 0, axis=0)
    return inside


class GridWorld:
    def __init__(self, shape, step: float = 0.5, density=1.0):
        self.density = density
        self.weights = None
        self.step = step
        self.shape = shape

        if isinstance(shape, Polygon):
            self.xmin, self.ymin, self.xmax, self.ymax = shape.bounds
            self.xmin = math.floor(self.xmin / step) * step
            self.ymin = math.floor(self.ymin / step) * step
            self.xmax = math.ceil(self.xmax / step) * step
            self.ymax = math.ceil(self.ymax / step) * step
            self.x_lim = (self.xmin / self.step, self.xmax / self.step)
            self.y_lim = (self.ymin / self.step, self.ymax / self.step)
            self.x_num = int((self.xmax - self.xmin) / step) + 1
            self.y_num = int((self.ymax - self.ymin) / step) + 1
            self.X, self.Y = np.meshgrid(np.arange(self.xmin, self.xmax + 0.1, step),
                                         np.arange(self.ymin, self.ymax + 0.1, step))
            assert self.X.shape == (self.y_num, self.x_num)
            self.weights = np.ones_like(self.X).astype(float) * density

        else:
            raise ValueError("Invalid shape type")

    def get_bound_inside_world(self, shape):
        xmin, ymin, xmax, ymax = shape.bounds
        xmin = max(xmin, self.xmin)
        ymin = max(ymin, self.ymin)
        xmax = min(xmax, self.xmax)
        ymax = min(ymax, self.ymax)
        return xmin, ymin, xmax, ymax

    def minus_density_in_shape(self, shape, density):
        dict = {'x': [], 'y': [], 'weight': []}
        xmin, ymin, xmax, ymax = self.get_bound_inside_world(shape)
        poly = np.array(shape.exterior.coords.xy).T
        mask_bbox = (self.X >= xmin) & (self.X <= xmax) & (self.Y >= ymin) & (self.Y <= ymax)
        points_in_bbox = np.column_stack([self.X[mask_bbox], self.Y[mask_bbox]])
        mask_polygon = is_point_in_polygon(points_in_bbox, poly)
        mask_polygon_2d = np.zeros_like(mask_bbox, dtype=bool)
        mask_polygon_2d[mask_bbox] = mask_polygon.reshape(-1)
        self.weights[mask_polygon_2d] -= density
        self.weights = np.clip(self.weights, 0, 1)
        dict['x'] = np.where(mask_polygon_2d)[1].tolist()
        dict['y'] = np.where(mask_polygon_2d)[0].tolist()
        dict['weight'] = self.weights[mask_polygon_2d]
        return dict

    def get_weighted_mean_point_in_shape(self, shape):
        xmin, ymin, xmax, ymax = self.get_bound_inside_world(shape)
        poly = np.array(shape.exterior.coords.xy).T
        mask_bbox = (self.X >= xmin) & (self.X <= xmax) & (self.Y >= ymin) & (self.Y <= ymax)
        points_in_bbox = np.column_stack([self.X[mask_bbox], self.Y[mask_bbox]])
        mask_polygon = is_point_in_polygon(points_in_bbox, poly)
        mask_polygon_2d = np.zeros_like(mask_bbox, dtype=bool)
        mask_polygon_2d[mask_bbox] = mask_polygon.reshape(-1)
        weights = self.weights[mask_polygon_2d]
        if np.any(mask_polygon):
            if np.sum(weights) == 0:
                mean_point = np.average(points_in_bbox[mask_polygon], axis=0)
            else:
                mean_point = np.average(points_in_bbox[mask_polygon], axis=0, weights=weights)
            return Point(mean_point)
        else:
            return Point(np.average(poly, axis=0))

    def output_gridworld(self, points):
        # print(self.weights)
        str_gridworld = np.where(self.weights >= 0.5, '.', ' ')
        for point in points:
            str_gridworld[int((point.y - self.ymin) / self.step), int((point.x - self.xmin) / self.step)] = 'o'
        output_ls = np.flipud(str_gridworld).tolist()
        # output output_array in line
        for line in output_ls:
            print(''.join(line))


