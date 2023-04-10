from shapely.geometry import Polygon, Point
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np


def cal_cvt(points, world):
    points = [Point(p) for p in points]
    # points = [p for p in points if world.contains(p)]
    points = [p.coords[0] for p in points]
    points = np.array(points)
    # print('original points', points)
    bbox = np.array(world.bounds).reshape(2, 2).T
    x_min, x_max = bbox[0]
    y_min, y_max = bbox[1]
    x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2
    x_left, x_right = x_mid - (x_max - x_min) * 2, x_mid + (x_max - x_min) * 2
    y_down, y_up = y_mid - (y_max - y_min) * 2, y_mid + (y_max - y_min) * 2
    points = np.concatenate((points, [
        [x_left, y_mid],
        [x_right, y_mid],
        [x_mid, y_down],
        [x_mid, y_up]
    ]))
    vor = Voronoi(points)
    regions, vertices, point_regions = vor.regions, vor.vertices, vor.point_region
    polygons = []
    for pr in point_regions:
        if -1 in regions[pr]:
            continue
        v = []
        for pt_ind in regions[pr]:
            v.append(vertices[pt_ind])
        v = np.array(v)
        poly = Polygon(v)
        # poly = poly.intersection(world)
        polygons.append(poly)
    if len(polygons) != len(points) - 4:
        from matplotlib import pyplot as plt

        # plot voronoi results
        voronoi_plot_2d(vor)
        plt.show()

        assert len(polygons) == len(points) - 4
    # print('')
    # for ind in range(len(points) - 4):
    #     print(f'Original point #{ind}: ', points[ind])
    #     print('Polygon: ', polygons[ind])
    return polygons
