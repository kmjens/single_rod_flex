import itertools
import signac
import numpy as np

def grid(gridspec):
    '''
    Input: dictionary whose keys correspond to parameter names.
    Keys are associated with iterable values that each parameter could be.
    Outputs: sequence of dictionaries where each has unique combo of parameter values.
    '''
    for values in itertools.product(*gridspec.values()):
        yield dict(zip(gridspec.keys(), values))

gridspec = {
    "N_mesh":   [1e3],
    "R":        [8], # Radius of spherical mesh
    "kT":       [0.2],
    "gamma":    [50],
    "k_bend":   [800],
    "k_area":   [5000],
    "dt":       [0.0005],
    "N_active": [1],
    "ratio_len":[2], # [2, 3, 4], # ratio of mesh diam to rod length (decides rod const. particle)
    "v0":       [40],
    "runtime":  [2e5], #[5e6],
    "equiltime": [1e4], #[1e5],
    "rod_size_int": [5], # rod length in units of recalculated sigma
    
    # To flatten (approx cylinders)
    "num_filler":  [20], #[0,10], #number of filler particles to flatten spherocylinder
    "filler_diam_ratio": [0.25], # ratio of filler particle diam to A diam 

    # To add gravity
    "gravity_strength": [1.0], # abs value gravity strength
    "gravity_ratio": [1], # how much strong grav on A beads than mesh
    
    # To make chiral:
    "rand_orient":  ['False'],
    "active_angle": [0], #angle in degrees
    "torque_mag":   [0],
}

if __name__ == "__main__":
    project = signac.init_project()
    for i, statepoint in enumerate(grid(gridspec)):
        statepoint["seed"] = i
        print("Opening job", statepoint)
        job = project.open_job(statepoint).init()
