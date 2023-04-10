import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point
import sys
import time
import json

import CBFSlack
import World
import GridWorld
import Robot
import CVT
import Swarm
import draw


def PreParse(launch_json):
    world = World.World(
        Polygon(
            [(p['x'], p['y']) for p in launch_json['world']]
        ).convex_hull,
        [Point(p['x'], p['y']) for p in launch_json['charge']]
    )
    swarm = Swarm.Swarm(launch_json['swarm']['num'], world)
    if launch_json['swarm']['initial_position'] == 'random_all':
        swarm.random_initial_position_in_shape(world.w)
    elif launch_json['swarm']['initial_position'] == 'random_in_poly':
        swarm.random_initial_position_in_shape(
            Polygon(
                [(p['x'], p['y']) for p in launch_json['swarm']['random_poly']]
            ).convex_hull
        )
    elif launch_json['swarm']['initial_position'] == 'specified':
        # check if the number of specified position is equal to the number of robots
        if len(launch_json['swarm']['specified_pos']) != launch_json['swarm']['num']:
            raise ValueError('The number of specified position is not equal to the number of robots')
        for i, robot in enumerate(swarm.robots):
            robot.set_position(Point(
                launch_json['swarm']['specified_pos'][i]['x'],
                launch_json['swarm']['specified_pos'][i]['y']
            ))
    cbf_options = {}
    for key, value in launch_json['cbfs'].items():
        if value == 'on':
            cbf_options[key] = True
        elif value == 'off':
            cbf_options[key] = False
        else:
            cbf_options[key] = value
    exec_options = launch_json['execute']
    return swarm, cbf_options, exec_options


if __name__ == '__main__':
    print('Welcome to cbf_py @xirhxq')
    # print python version
    print('Python version: ', sys.version)
    tic = time.time()

    # parse launch_json from 'cbf_main.json' in /launch
    with open('launch/cbf_main.json', 'r') as f:
        launch_json = json.load(f)

    swarm, cbf_options, exec_options = PreParse(launch_json)
    if cbf_options['energy_cbf']:
        swarm.set_energy_cbf()
    swarm.para_log()
    t_total, t_gap = exec_options['time_total'], exec_options['step_time']
    pb = draw.MyProgressBar(t_total)
    for t in np.arange(0, t_total, t_gap):
        pb.update(t)
        # print('Time: ', t)
        # swarm.gridworld.output_gridworld()
        # for robot in swarm.robots:
        #     robot.output_state()
        if cbf_options['cvt_cbf']:
            swarm.set_cvt_cbf()
        if cbf_options['comm_cbf']:
            swarm.set_comm_cbf(cbf_options['comm_order'])
        swarm.time_forward(t_gap)
        swarm.update_gridworld()
        swarm.log_once()
        # plt.Figure()
        # swarm.gridworld.draw_gridworld_with_matplotlib_with_density()
        # swarm.draw_cvt()
        # swarm.draw_position()
    pb.end()
    swarm.save_log()
    toc = time.time()
    print('Time cost: ', toc - tic)
    draw.menu()
