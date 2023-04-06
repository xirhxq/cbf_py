import math
import numpy as np
from shapely.geometry import Polygon, Point
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


class GridWorld:
    def __init__(self, shape, step=1, density=1.0):
        self.density = density
        self.points = []
        self.weights = []
        self.step = step

        if isinstance(shape, Polygon):
            xmin, ymin, xmax, ymax = shape.bounds
            for i in range(math.ceil(xmin / step), math.floor(xmax / step) + 1):
                for j in range(math.ceil(ymin / step), math.floor(ymax / step) + 1):
                    point = Point(i * step, j * step)
                    if shape.contains(point):
                        self.points.append([i * step, j * step])
                        self.weights.append(density)
        else:
            raise ValueError("Invalid shape type")

        self.points = np.array(self.points)
        self.weights = np.array(self.weights)

    def set_density_in_shape(self, shape, density):
        xmin, ymin, xmax, ymax = shape.bounds
        for i in range(math.ceil(xmin / self.step), math.floor(xmax / self.step) + 1):
            for j in range(math.ceil(ymin / self.step), math.floor(ymax / self.step) + 1):
                point = Point(i * self.step, j * self.step)
                if shape.contains(point):
                    self.weights[i] = density

    def minus_density_in_shape(self, shape, density):
        xmin, ymin, xmax, ymax = shape.bounds
        for i in range(math.ceil(xmin / self.step), math.floor(xmax / self.step) + 1):
            for j in range(math.ceil(ymin / self.step), math.floor(ymax / self.step) + 1):
                point = Point(i * self.step, j * self.step)
                if shape.contains(point):
                    self.weights[i] -= density
                    if self.weights[i] < 0:
                        self.weights[i] = 0

    def get_weighted_mean_point_in_shape(self, shape):
        total_weighted_sum = np.array([0, 0]).astype(float)
        total_weight = 0
        xmin, ymin, xmax, ymax = shape.bounds
        for i in range(math.ceil(xmin / self.step), math.floor(xmax / self.step) + 1):
            for j in range(math.ceil(ymin / self.step), math.floor(ymax / self.step) + 1):
                point = Point(i * self.step, j * self.step)
                if shape.contains(point):
                    total_weighted_sum += self.weights[i] * self.points[i]
                    total_weight += self.weights[i]
        if total_weight == 0:
            return shape.centroid.x, shape.centroid.y
        return Point(total_weighted_sum / total_weight)

    def draw_gridworld_with_matplotlib(self):
        plt.scatter(self.points[:, 0], self.points[:, 1], s=1)
        plt.show()

    def draw_gridworld_with_matplotlib_with_density(self):
        plt.scatter(self.points[:, 0], self.points[:, 1], s=1, c=self.weights)
        plt.colorbar()
        plt.clim(0, 1)
        plt.show()