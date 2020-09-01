import numpy as np

MAXIMUM_RAM = 1 * 2 ** 30  # Set maximum of 1 GB RAM
V_TYPE = np.float32
V_TYPE_SIZE = 8

from gym_mapf.solvers.id import id
from gym_mapf.solvers.vi import value_iteration, prioritized_value_iteration
from gym_mapf.solvers.pi import policy_iteration
from gym_mapf.solvers.rtdp import (rtdp_iterations_generator,
                                   fixed_iterations_count_rtdp,
                                   stop_when_no_improvement_between_batches_rtdp)
from gym_mapf.solvers.lrtdp import lrtdp
