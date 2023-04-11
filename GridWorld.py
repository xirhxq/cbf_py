import math
import numpy as np
from shapely.geometry import Polygon, Point, LineString
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


class GridWorld:
    def __init__(self, shape, step: float = 0.5, density=1.0):
        self.density = density
        self.weights = []
        self.step = step
        self.shape = shape

        if isinstance(shape, Polygon):
            xmin, ymin, xmax, ymax = shape.bounds
            self.x_num = math.floor(xmax / step) - math.ceil(xmin / step) + 1
            self.y_num = math.floor(ymax / step) - math.ceil(ymin / step) + 1
            self.x_lim = (math.ceil(xmin / step), math.floor(xmax / step))
            self.y_lim = (math.ceil(ymin / step), math.floor(ymax / step))
            self.weights = np.ones((self.x_num, self.y_num)).astype(float) * density
        else:
            raise ValueError("Invalid shape type")

    def get_xy_index_from_point(self, point):
        return (int(point.x / self.step) - self.x_lim[0],
                int(point.y / self.step) - self.y_lim[0])

    def get_bound_inside_world(self, shape):
        xmin, ymin, xmax, ymax = shape.bounds
        xmin = max(xmin, self.x_lim[0] * self.step)
        ymin = max(ymin, self.y_lim[0] * self.step)
        xmax = min(xmax, self.x_lim[1] * self.step)
        ymax = min(ymax, self.y_lim[1] * self.step)
        return xmin, ymin, xmax, ymax

    # def minus_density_in_shape_slow(self, shape, density):
    #     xmin, ymin, xmax, ymax = self.get_bound_inside_world(shape)
    #     dict = []
    #     for i in range(math.ceil(xmin / self.step), math.floor(xmax / self.step) + 1):
    #         for j in range(math.ceil(ymin / self.step), math.floor(ymax / self.step) + 1):
    #             point = Point(i * self.step, j * self.step)
    #             if shape.contains(point):
    #                 xy_index = self.get_xy_index_from_point(point)
    #                 self.weights[xy_index] -= density
    #                 dict.append({'x': xy_index[0], 'y': xy_index[1], 'weight': self.weights[xy_index]})
    #                 if self.weights[xy_index] < 0:
    #                     self.weights[xy_index] = 0
    #     return dict

    def minus_density_in_shape(self, shape, density):
        xmin, ymin, xmax, ymax = self.get_bound_inside_world(shape)
        dict = []
        for i in range(math.ceil(xmin / self.step), math.floor(xmax / self.step) + 1):
            y_line = shape.intersection(LineString([(i * self.step, ymin), (i * self.step, ymax)]))
            if y_line.is_empty:
                raise ValueError("y_line is empty")
            ydown, yup = y_line.bounds[1], y_line.bounds[3]
            for j in range(math.ceil(ydown / self.step), math.floor(yup / self.step) + 1):
                point = Point(i * self.step, j * self.step)
                xy_index = self.get_xy_index_from_point(point)
                self.weights[xy_index] -= density
                dict.append({'x': xy_index[0], 'y': xy_index[1], 'weight': self.weights[xy_index]})
                if self.weights[xy_index] < 0:
                    self.weights[xy_index] = 0
        return dict

    def get_weighted_mean_point_in_shape(self, shape):
        total_weighted_sum = np.array([0, 0]).astype(float)
        total_weight = 0
        xmin, ymin, xmax, ymax = self.get_bound_inside_world(shape)
        for i in range(math.ceil(xmin / self.step), math.floor(xmax / self.step) + 1):
            y_line = shape.intersection(LineString([(i * self.step, ymin), (i * self.step, ymax)]))
            if y_line.is_empty:
                continue
            ydown, yup = y_line.bounds[1], y_line.bounds[3]
            for j in range(math.ceil(ydown / self.step), math.floor(yup / self.step) + 1):
                point = Point(i * self.step, j * self.step)
                xy_index = self.get_xy_index_from_point(point)
                total_weighted_sum += self.weights[xy_index] * np.array([i * self.step, j * self.step])
                total_weight += self.weights[xy_index]
        if total_weight == 0:
            shape = shape.intersection(self.shape)
            return Point(shape.centroid.x, shape.centroid.y)
        return Point(total_weighted_sum / total_weight)

    def get_cvt_cost_in_shape(self, shape, point):
        res = 0
        xmin, ymin, xmax, ymax = self.get_bound_inside_world(shape)
        for i in range(math.ceil(xmin / self.step), math.floor(xmax / self.step) + 1):
            y_line = shape.intersection(LineString([(i * self.step, ymin), (i * self.step, ymax)]))
            if y_line.is_empty:
                continue
            ydown, yup = y_line.bounds[1], y_line.bounds[3]
            for j in range(math.ceil(ydown / self.step), math.floor(yup / self.step) + 1):
                xy_index = self.get_xy_index_from_point(Point(i * self.step, j * self.step))
                res += self.weights[xy_index] * math.sqrt((i * self.step - point.x) ** 2 + (j * self.step - point.y) ** 2)
        return res

    def draw_gridworld_with_matplotlib_with_density(self):
        plt.imshow(self.weights.T, alpha=0.2, interpolation='nearest', extent=(self.x_lim[0] * self.step,
                                                                               self.x_lim[1] * self.step,
                                                                               self.y_lim[0] * self.step,
                                                                               self.y_lim[1] * self.step),
                   origin='lower')
        plt.colorbar()
        # set cbar from 0 to 1
        plt.show()

    def output_gridworld(self):
        # output gridworld directly to terminal
        for i in range(self.y_num):
            for j in range(self.x_num):
                # map self.weights[j, i] to 0-9
                print(int(self.weights[j, i] * 9), end='')
            print('')
