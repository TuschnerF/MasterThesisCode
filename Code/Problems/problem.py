import numpy as np

class Problem:
    # Constructor
    def __init__(self):
        pass

    # M
    def imaging_operator(self):
        print("Test superklasse")

    #M^\star
    def back_projection(self):
        pass

    # K_varphi
    def cutoff_operator(self):
        pass

    # P
    def pseudo_operator(self):
        pass

    # M^starPM  or M^starPM 
    def reconstruction_operator(self):
        pass


def solve_problem(problem_class: object, path):
    instance_prob = problem_class(path)
    instance_prob.run()


# def load_data(self, path: str|Path) -> np.ndarray:
#     pass

def plot_data(self):
    pass