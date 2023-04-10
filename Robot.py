import numpy as np
from shapely.geometry import Point
import json
import gurobipy as gp
from gurobipy import GRB

import World
import CBFMultiNoSlack


class Robot:
    def __init__(self, dim_x, dim_u, x_ord=0, y_ord=1):
        self.G = np.zeros((dim_x, dim_u)).astype(float)
        self.X = np.zeros((dim_x, 1)).astype(float)
        self.F = np.zeros((dim_x, 1)).astype(float)
        self.U = np.zeros((dim_u, 1))
        self.x_ord = x_ord
        self.G[x_ord, x_ord] = 1
        self.y_ord = y_ord
        self.G[y_ord, y_ord] = 1
        self.batt_ord = 2
        self.camera_ord = 3
        self.cbf_slack = {}
        self.cbf_no_slack = CBFMultiNoSlack.CBFMultiNoSlack('Energy&Safe', {})

    def output_FGU(self):
        print(f'F{self.F.size}: ', self.F)
        print(f'G{self.G.size}: ', self.G)
        print(f'U{self.U.size}: ', self.U)

    def output_state(self):
        print('x: ', self.X[self.x_ord])
        print('y: ', self.X[self.y_ord])
        print('batt: ', self.X[self.batt_ord])

    def xy(self):
        return Point(self.X[self.x_ord], self.X[self.y_ord])

    def set_position(self, point):
        self.X[self.x_ord] = point.x
        self.X[self.y_ord] = point.y

    def set_battery(self, batt):
        if self.X.size > self.batt_ord:
            self.X[self.batt_ord] = batt

    def xy(self) -> Point:
        return Point(self.X[self.x_ord], self.X[self.y_ord])

    def batt(self):
        return float(self.X[self.batt_ord])

    def time_forward(self, runtime, dt, world: World):
        if world.is_charging(self.xy()) and self.X[self.batt_ord] < 100:
            self.X[self.batt_ord] += 10 * dt
        else:
            opt_res = np.zeros((self.U.size, 1))
            model = gp.Model("model")
            model.setParam('OutputFlag', 0)

            var_v = []
            for i in range(self.U.size):
                var_v.append(model.addVar(-gp.GRB.INFINITY, gp.GRB.INFINITY, 0.0, gp.GRB.CONTINUOUS, f'var_{i}'))

            var_slack = []
            for i in range(len(self.cbf_slack)):
                var_slack.append(
                    model.addVar(-gp.GRB.INFINITY, gp.GRB.INFINITY, 0.0, gp.GRB.CONTINUOUS, f'slack_{i}'))

            model.setObjective(sum([i * i for i in var_v]) + 0.1 * sum([i * i for i in var_slack]), gp.GRB.MINIMIZE)

            u_coe = self.cbf_no_slack.constraint_u_coe(self.F, self.G, self.X, runtime)
            constraint_const = self.cbf_no_slack.constraint_const_with_time(self.F, self.G, self.X, runtime)
            # print(f'cbf_{name}: ', u_coe, constraint_const)
            # print(f'cbf_{name} expr: ', (u_coe.dot(var_v) + constraint_const)[0, 0])
            lhs = u_coe.dot(var_v) + constraint_const

            # if isinstance(lhs, np.ndarray):
            #     lhs = lhs[0, 0]

            model.addConstr(lhs >= 0, name=f'cbf_{self.cbf_no_slack.name}')

            cnt = 0
            for name, cbf in self.cbf_slack.items():
                u_coe = cbf.constraint_u_coe(self.F, self.G, self.X, runtime)
                constraint_const = cbf.constraint_const_without_time(self.F, self.G, self.X, runtime)
                # print(f'cbf_slack_{name}: ', u_coe, constraint_const)
                # print(f'cbf_slack_{name} expr: ', u_coe.dot(var_v) + var_slack[cnt] + constraint_const)
                model.addConstr(u_coe.dot(var_v) + var_slack[cnt] + constraint_const >= 0, name=f'cbf_slack_{name}')
                cnt += 1

            model.optimize()

            for i in range(self.U.size):
                opt_res[i] = var_v[i].x

            slack_res = np.zeros(len(var_slack))
            for i in range(len(var_slack)):
                slack_res[i] = var_slack[i].x

            # print('opt_res: ', opt_res)
            # print('slack_res: ', slack_res)

            self.X += (self.F + self.G @ opt_res) * dt