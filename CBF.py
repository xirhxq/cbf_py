import numpy as np


class CBF:
    def __init__(self, name, h, alpha=lambda h: 0.1 * np.power(h, 3), delta=0.001):
        self.name = name
        self.delta = delta
        self.alpha = alpha
        self.h = h

    def dhdxi(self, x, t, i):
        x_plus_dx = x.copy()
        x_plus_dx[i] += self.delta
        x_minus_dx = x.copy()
        x_minus_dx[i] -= self.delta
        return (self.h(x_plus_dx, t) - self.h(x_minus_dx, t)) / (2 * self.delta)

    def dhdt(self, x, t):
        return (self.h(x, t + self.delta) - self.h(x, t - self.delta)) / (2 * self.delta)

    def dhdx(self, x, t):
        return np.array([self.dhdxi(x, t, i) for i in range(len(x))])

    def constraint_u_coe(self, f, g, x, t):
        # print('dhdx: ', self.dhdx(x, t).T)
        # print('g: ', g)
        return self.dhdx(x, t).T @ g

    def constraint_const_with_time(self, f, g, x, t):
        # print('dhdt: ', self.dhdt(x, t))
        # print('dhdx: ', self.dhdx(x, t).T)
        # print('f: ', f)
        # print('alpha: ', self.alpha(self.h(x, t)))
        return self.dhdt(x, t) + self.dhdx(x, t).T @ f + self.alpha(self.h(x, t))

    def constraint_const_without_time(self, f, g, x, t):
        return self.dhdx(x, t).T @ f + self.alpha(self.h(x, t))