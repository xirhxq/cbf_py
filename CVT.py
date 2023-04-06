from shapely.geometry import Polygon, Point
import pyvoro
import numpy as np
from scipy.spatial import Voronoi


def cal_cvt(points, world):
    points = [Point(p) for p in points]
    points = [p for p in points if world.contains(p)]
    points = [p.coords[0] for p in points]
    points = np.array(points)
    bbox = np.array(world.bounds).reshape(2, 2).T
    cells = pyvoro.compute_2d_voronoi(points, bbox, 0.1)
    vertices = [cell['vertices'] for cell in cells]
    polygons = []
    for v in vertices:
        polygons.append(Polygon(v))
    # polygons = [p.intersection(world) for p in polygons if world.contains(p)]
    return polygons


def voronoi_polygons_scipy(points, world):
    points = [Point(p) for p in points]
    points = [p for p in points if world.contains(p)]
    points = [p.coords[0] for p in points]
    points = np.array(points)
    vor = Voronoi(points)
    print(vor.regions)
    polygons = []
    for region in vor.regions:
        if -1 not in region:
            polygon = Polygon([vor.vertices[i] for i in region])
            polygons.append(polygon.intersection(world))
    return polygons
