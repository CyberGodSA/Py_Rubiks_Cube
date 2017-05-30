import pycuber as pc
from pycuber.solver import CFOPSolver


def cube_solver_():
    formula = ""
    moves = [('R', 1, 0), ('B', 2, 0), ('D', -1, 0), ('R', -1, 0), ('B', 1, 0), ('R', 1, 0), ('U', -1, 0)]
    for i in moves:
        formula += i[0]
        if i[1] == -1:
            formula += "'"
        elif i[1] == 2:
            formula += "2"
        formula += " "
    print(moves)
    c = pc.Cube()
    my_formula = pc.Formula(formula)
    c(my_formula)
    solution = CFOPSolver(c).solve(suppress_progress_messages=True).__str__()
    print(my_formula)
    print(solution)
    move_list = []
    for i in solution.split(" "):
        if len(i) == 2:
            if i[1] == "'":
                move_list.append((i[0], -1, 0))
            elif i[1] == "2":
                move_list.append((i[0], 2, 0))
        else:
            move_list.append((i[0], 1, 0))
    print(move_list)
    return move_list


if __name__ == "__main__":
    cube_solver_()
