import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point

import CBF
import World
import GridWorld
import Robot
import CVT
import Swarm

def test_CBF():
    cbf = CBF.CBF(lambda x, t: x[0] + x[1] + x[2] + t)

    f = np.array([[1, 2, 3]]).T.astype(float)
    g = np.array([[4, 5], [6, 7], [8, 9]]).astype(float)
    x = np.array([[6, 7, 8]]).T.astype(float)
    t = 1.0

    print('ucoe = ', cbf.constraint_u_coe(f, g, x, t))
    print('const with time', cbf.constraint_const_with_time(f, g, x, t))
    print('const w/o time', cbf.constraint_const_without_time(f, g, x, t))


def test_World():
    w = World.World(Polygon([(0, 0), (0, 10), (10, 10), (5, 0)]), [(1, 1), (2, 2), (3, 3)])
    w.output_world()
    print(w.get_random_point())
    w.draw_polygon_with_matplotlib()


def test_GridWorld():
    world_poly = Polygon([(-15, 0), (15, 0), (-15, 30), (15, 30)]).convex_hull
    gw = GridWorld.GridWorld(world_poly)
    gw.draw_gridworld_with_matplotlib_with_density()
    world = World.World(world_poly, [])
    for i in range(3):
        random_points = [world.get_random_point() for _ in range(4)]
        random_poly = Polygon(random_points).convex_hull
        print(random_poly.exterior.coords.xy)
        import matplotlib.pyplot as plt
        plt.plot(*random_poly.centroid.xy, 'bo')
        print('shape centroid', *random_poly.centroid.xy)
        print('weighted centroid', gw.get_weighted_mean_point_in_shape(random_poly))
        plt.plot(*gw.get_weighted_mean_point_in_shape(random_poly), 'ro')
        gw.draw_gridworld_with_matplotlib_with_density()
        gw.minus_density_in_shape(random_poly, 1.0)
    gw.draw_gridworld_with_matplotlib_with_density()

    for i in range(3):
        # get random circle inside world_poly
        random_point = world.get_random_point()
        random_radius = np.random.uniform(0, 5)
        random_circle = random_point.buffer(random_radius)
        print('shape centroid', *random_circle.centroid.xy)
        print('weighted centroid', gw.get_weighted_mean_point_in_shape(random_circle))
        import matplotlib.pyplot as plt
        plt.plot(*random_circle.centroid.xy, 'bo')
        plt.plot(*gw.get_weighted_mean_point_in_shape(random_circle), 'ro')
        gw.minus_density_in_shape(random_circle, 1.0)
        gw.draw_gridworld_with_matplotlib_with_density()
    gw.draw_gridworld_with_matplotlib_with_density()


def test_Robot():
    robot = Robot.Robot(3, 2)
    robot.output_state()
    world_poly = Polygon([(-15, 0), (15, 0), (-15, 30), (15, 30)]).convex_hull
    world = World.World(world_poly, [])
    robot.time_forward(0.0, 0.02, world)
    robot.output_state()


def test_CVT():
    world_poly = Polygon([(-15, 0), (15, 0), (-15, 30), (15, 30)]).convex_hull
    world = World.World(world_poly, [])
    random_points = [world.get_random_point() for _ in range(5)]
    cells = CVT.cal_cvt(random_points, world.w)
    import matplotlib.pyplot as plt
    for cell in cells:
        plt.plot(*cell.exterior.xy, 'r')
    for p in random_points:
        plt.plot(*p.xy, 'bo')
    plt.show()


def test_Swarm():
    world_poly = Polygon([(-15, 0), (15, 0), (-15, 30), (15, 30)]).convex_hull
    world = World.World(world_poly, [])
    swarm = Swarm.Swarm(2, world)


if __name__ == '__main__':
    print('Welcome to cbf_py @xirhxq')
    world_poly = Polygon([(-15, 0), (15, 0), (-15, 30), (15, 30)]).convex_hull
    world = World.World(world_poly, [Point([0.0, 0.0])])
    swarm = Swarm.Swarm(22, world)
    swarm.set_energy_cbf()
    t_total, t_gap = 1, 0.02
    for t in np.arange(0, t_total, t_gap):
        # for robot in swarm.robots:
        #     robot.output_state()
        swarm.set_cvt_cbf()
        swarm.time_forward(t_gap)
        swarm.update_gridworld()
        # plt.Figure()
        # swarm.gridworld.draw_gridworld_with_matplotlib_with_density()
        # set x and y limit
        # plt.xlim(-15, 15)
        # plt.ylim(0, 30)



