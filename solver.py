import pycuber as pc
from pycuber.solver import CFOPSolver

import numpy as np

from cube import *


def cube_solver():
    formula = ""
    #moves = cube._move_list
    moves = [('R', 1, 0), ('B', 2, 0), ('D', -1, 0), ('R', -1, 0), ('B', 1, 0), ('R', 1, 0), ('U', -1, 0)]
    for i in moves:
        formula += i[0]
        if i[1] == -1:
            formula += "'"
        if i[1] == 2:
            formula += "2"
        formula += " "

    print(formula)

    c = pc.Cube()
    my_formula = pc.Formula(formula)
    c(my_formula)
    solver = CFOPSolver(c)
    solution = solver.solve(suppress_progress_messages=True)

    print(solution)


cube_solver()
