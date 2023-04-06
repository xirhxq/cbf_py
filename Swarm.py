from math import log
from shapely.geometry import Point
import time

import Robot, World, GridWorld, CVT, CBF


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'Function {func.__name__} run time: {end_time - start_time} s')
        return result
    return wrapper


class Swarm:
    def __init__(self, n, world: World):
        self.robots = [Robot.Robot(3, 2) for _ in range(n)]
        for robot in self.robots:
            robot.set_position(world.get_random_point())
            robot.F[2] = -1
            robot.X[2] = 100
        self.world = world
        self.gridworld = GridWorld.GridWorld(world.w)
        self.runtime = 0.0

    def set_energy_cbf(self):
        for robot in self.robots:
            def battery_h(x, t):
                xy = Point(x[robot.x_ord], x[robot.y_ord])
                nearest_c = self.world.nearest_charge_place(xy)
                lg = log(nearest_c[0].distance(xy) / nearest_c[1])
                return x[robot.batt_ord] - lg

            def energy_h(x, t):
                return battery_h(x, t)
            robot.cbf_no_slack['energy'] = CBF.CBF('energy', energy_h)

    def set_cvt_cbf(self):
        points = [robot.xy() for robot in self.robots]
        # for index, p in enumerate(points):
        #     print(f'Robot #{index}', p)
        # print('World: ', self.world.w)
        cells = CVT.cal_cvt(points, self.world.w)
        centers = [self.gridworld.get_weighted_mean_point_in_shape(cell) for cell in cells]
        for index, robot in enumerate(self.robots):
            def cvt_h(x, t):
                xy = Point(x[robot.x_ord], x[robot.y_ord])
                k_cvt = 5.0
                return -k_cvt * xy.distance(centers[index])
            robot.cbf_slack['cvt'] = CBF.CBF('cvt', cvt_h, alpha=lambda h: h)

    def time_forward(self, dt):
        for robot in self.robots:
            robot.time_forward(self.runtime, dt, self.world)
        self.runtime += dt

    def update_gridworld(self):
        for robot in self.robots:
            self.gridworld.minus_density_in_shape(robot.xy().buffer(1.0), 0.2)


