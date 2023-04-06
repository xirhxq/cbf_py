from shapely.geometry import Polygon, Point
import random


class World:
    def __init__(self, w: Polygon, charging_stations: list[Point]):
        self.w = w
        self.c = [(c, 0.3) for c in charging_stations]

    def get_random_point(self):
        minx, miny, maxx, maxy = self.w.bounds
        while True:
            p = Point([random.uniform(minx, maxx), random.uniform(miny, maxy)])
            if self.w.contains(p):
                return p

    def output_world(self):
        print(self.w.exterior.coords.xy)
        print(self.c)

    def draw_polygon_with_matplotlib(self):
        import matplotlib.pyplot as plt
        x, y = self.w.exterior.coords.xy
        plt.plot(x, y)
        # draw charging stations
        for c in self.c:
            plt.plot(c[0][0], c[0][1], 'ro')
        plt.show()

    def draw_polygon_with_matplotlib_with_charging_stations(self):
        import matplotlib.pyplot as plt
        x, y = self.w.exterior.coords.xy
        plt.plot(x, y)
        # draw charging stations
        for c in self.c:
            plt.plot(c[0][0], c[0][1], 'ro')
        plt.show()

    def nearest_charge_place(self, p: Point):
        if len(self.c) == 0:
            return None
        return min(self.c, key=lambda x: x[0].distance(p))

    def dist_to_charge_place(self, p: Point):
        if len(self.c) == 0:
            return None
        return self.nearest_charge_place(p)[0].distance(p)

    def is_charging(self, p: Point):
        if len(self.c) == 0:
            return False
        return self.dist_to_charge_place(p) < self.nearest_charge_place(p)[1]





