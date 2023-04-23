from math import log, pi
from shapely.geometry import Point, Polygon
import time
import numpy as np

import Robot, World, GridWorld, CVT, CBFSlack


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'Function {func.__name__} run time: {end_time - start_time} s')
        return result

    return wrapper


class Swarm:
    def __init__(self, n, world: World, bases):
        self.update = None
        self.centers = None
        self.cells = None
        self.robots = [Robot.Robot(3, 2) for _ in range(n)]
        for robot in self.robots:
            robot.F[2] = -1
            # x[2] = random 10 to 20
            robot.X[2] = 10 + 10 * np.random.random()
        self.world = world
        self.gridworld = GridWorld.GridWorld(world.w)
        self.runtime = 0.0
        self.data = {'state': []}
        self.log_name = time.strftime("%m-%d_%H-%M", time.localtime())
        self.bases = bases

    def random_initial_position_in_shape(self, shape):
        for robot in self.robots:
            robot.set_position(self.world.get_random_point_in_shape(shape))

    def set_energy_cbf(self):
        for robot in self.robots:
            def battery_h(x, t, r=robot):
                xy = Point(x[r.x_ord], x[r.y_ord])
                nearest_c = self.world.nearest_charge_place(xy)
                lg = log(nearest_c[0].distance(xy) / nearest_c[1])
                return float(x[robot.batt_ord] - lg)

            def energy_h(x, t):
                return battery_h(x, t)

            # robot.cbf_no_slack['energy'] = CBFSlack.CBFSlack('energy', energy_h)
            robot.cbf_no_slack.add_h('energy', energy_h)

    # @timer
    def set_cvt_cbf(self):
        points = [robot.xy() for robot in self.robots]
        self.cells = CVT.cal_cvt(points, self.world.w)

        # get the corners of self.world.w in type of Point
        # corners = [Point(self.world.w.exterior.coords[i]) for i in range(4)]
        # for c in corners:
        #     # if c is not in any of the cells, raise an error
        #     if not any([c.within(cell) for cell in self.cells]):
        #         raise Exception(f'Corner {c} is not in any of the cells')

        self.centers = [self.gridworld.get_weighted_mean_point_in_shape(cell) for cell in self.cells]
        # print(self.centers, len(self.centers))
        for index, robot in enumerate(self.robots):
            def cvt_h(x, t, i=index, r=robot, c=self.centers[index]):
                xy = Point(x[r.x_ord], x[r.y_ord])
                k_cvt = 5.0
                # print(f'Robot #{i} @ {xy} Center @ {c} Distance: {xy.distance(c)}')
                return -k_cvt * xy.distance(c)

            robot.cbf_slack['cvt'] = CBFSlack.CBFSlack('cvt', cvt_h, alpha=lambda h: h)

            # def cvt_cost_h(x, t, i=index, r=robot, c=self.cells[index]):
            #     xy = Point(x[r.x_ord], x[r.y_ord])
            #     return -0.01 * self.gridworld.get_cvt_cost_in_shape(c, xy)

            # robot.cbf_slack['cvt_cost'] = CBFSlack.CBFSlack('cvt_cost', cvt_cost_h, alpha=lambda h: h)

    def set_comm_cbf(self, comm_order):
        comm_dis = comm_order['distance']
        if comm_order['type'] == 'fixed':
            for index, robot in enumerate(self.robots):
                for other in comm_order[str(index + 1)]:
                    if other[0] == 'b':
                        base_index = int(other[1:]) - 1
                        assert base_index < len(self.bases)
                        other_point = self.bases[base_index]
                        # other_point = self.world.c[base_index][0]
                    else:
                        other_index = int(other) - 1
                        other_point = self.robots[other_index].xy()

                    # print(f'{index + 1} {robot.xy()} to {other} @ {other_point}')

                    def comm_h(x, t, p=other_point, d=comm_dis, r=robot):
                        xy = Point(x[r.x_ord], x[r.y_ord])
                        k_comm = 1.0
                        return k_comm * (d - xy.distance(p))

                    robot.cbf_no_slack.add_h('comm', comm_h)

    # @timer
    def time_forward(self, dt):
        for robot in self.robots:
            robot.time_forward(self.runtime, dt, self.world)
        self.runtime += dt

    # @timer
    def update_gridworld(self):
        self.update = {'x': [], 'y': [], 'weight': []}
        for robot in self.robots:
            ret = self.gridworld.minus_density_in_shape(robot.xy().buffer(3.0), 1.0)
            self.update['x'].extend(ret['x'])
            self.update['y'].extend(ret['y'])
            self.update['weight'].extend(ret['weight'])

    def output_position(self):
        print('Time: ', self.runtime)
        for index, robot in enumerate(self.robots):
            print(f'Robot #{index}: ', robot.xy())

    def draw_position(self):
        import matplotlib.pyplot as plt
        for robot in self.robots:
            plt.plot(robot.X[robot.x_ord], robot.X[robot.y_ord], 'b*')

    def draw_cvt(self):
        import matplotlib.pyplot as plt
        for center in self.centers:
            plt.plot(center.x, center.y, 'ro')
        for cell in self.cells:
            # draw Polygon with matplotlib
            x, y = cell.exterior.xy
            plt.plot(x, y, color='#6699cc', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

    def para_log(self):
        minx, miny, maxx, maxy = self.world.w.bounds
        para_dict = {
            'number': len(self.robots), 'lim': {'x': [minx, maxx], 'y': [miny, maxy]},
            'world': [{'x': p[0], 'y': p[1]} for p in self.world.w.exterior.coords],
            'charge': {
                'num': len(self.world.c),
                'pos': [{
                    'x': p[0].x,
                    'y': p[0].y,
                } for p in self.world.c],
                'dist': [p[1] for p in self.world.c]
            },
            'grid_world': {'x_num': self.gridworld.x_num, 'y_num': self.gridworld.y_num,
                           'x_lim': [l * self.gridworld.step for l in self.gridworld.x_lim],
                           'y_lim': [l * self.gridworld.step for l in self.gridworld.y_lim]}
        }
        self.data['para'] = para_dict

    def log_once(self):
        state_dict = {'runtime': self.runtime,
                      'robot': [
                          {
                              'x': r.xy().x,
                              'y': r.xy().y,
                              'batt': r.batt(),
                              **{cbf.name: float(cbf.h(r.X, self.runtime)) for cbf in r.cbf_slack.values()},
                              **{name: float(h(r.X, self.runtime)) for name, h in r.cbf_no_slack.h_dict.items()},
                              'camera': pi
                          }
                          for r in self.robots
                      ],
                      'cvt': [
                          {
                              'num': len(cell.exterior.coords),
                              'pos': [{'x': p[0], 'y': p[1]} for p in cell.exterior.coords],
                              'center': {'x': self.centers[index].x, 'y': self.centers[index].y}
                          }
                          for index, cell in enumerate(self.cells)
                      ],
                      'update': self.update
                      }
        self.data['state'].append(state_dict)

    def save_log(self):
        import json
        import os
        if not os.path.exists('data'):
            os.mkdir('data')
        # print(self.data)
        with open(f'data/{self.log_name}_data.json', 'w') as f:
            json_data = json.dumps(self.data, indent=None)
            f.write(json_data)
            print(f'data saved to data/{self.log_name}_data.json')
