from pyomo.opt.results import SolverStatus
from pyomo.contrib.iis import write_iis
import pyomo.environ as pyo

from datetime import datetime


PYO_VAR_TYPE = pyo.NonNegativeIntegers
PYO_PARAM_TYPE = pyo.NonNegativeIntegers


class BaseAbstractModel:
    def __init__(self):
        self.model = pyo.AbstractModel()
        self.name = "BaseAbstractModel"

    # @abstractmethod
    def _provide_initial_solution(self, instance, initial_solution):
        pass

    def generate_instance(self, data: dict):
        return self.model.create_instance(data)

    def solve(self, instance, solver_options: dict, solver_name: str = "glpk", initial_solution: dict = None):
        # initialize solver and set options
        solver = pyo.SolverFactory(solver_name)
        for k, v in solver_options.items():
            solver.options[k] = v
        # provide initial solution (if any)
        # warmstart = False
        # if initial_solution is not None:
        #   instance = self._provide_initial_solution(instance, initial_solution)
        #   warmstart = True
        # solve
        s = datetime.now()
        results = solver.solve(instance)  # , warmstart = warmstart)
        e = datetime.now()
        # check solver status
        get_solution_ok = results.solver.status == SolverStatus.ok
        if not get_solution_ok:
            if (results.solver.status == SolverStatus.aborted) and (len(results.solution) > 0):
                get_solution_ok = True
        solution = {
            "solver_status": str(results.solver.status),
            "solution_exists": get_solution_ok,
            "termination_condition": str(results.solver.termination_condition),
            "runtime": (e - s).total_seconds(),
        }
        # get solution
        if get_solution_ok:
            # get solution
            for v in instance.component_objects(pyo.Var, active=True):
                solution[v.name] = []
                for idx in v:
                    solution[v.name].append(pyo.value(v[idx]))
            # get objective function value
            solution["obj"] = pyo.value(instance.OBJ)
        else:
            _ = write_iis(instance, "infeas.ilp", "gurobi")
        return solution


class BaseLoadManagementModel(BaseAbstractModel):
    def __init__(self):
        super().__init__()
        self.name = "BaseLoadManagementModel"
        ###########################################################################
        # Problem parameters
        ###########################################################################
        # number and set of nodes
        self.model.Nn = pyo.Param(within=pyo.NonNegativeIntegers)
        self.model.N = pyo.RangeSet(1, self.model.Nn)
        # number and set of functions
        self.model.Nf = pyo.Param(within=pyo.NonNegativeIntegers)
        self.model.F = pyo.RangeSet(1, self.model.Nf)
        # incoming load
        self.model.incoming_load = pyo.Param(self.model.N, self.model.F, within=PYO_PARAM_TYPE)
        # neighborhood (n_{ij}=1 if neighbors)
        self.model.neighborhood = pyo.Param(self.model.N, self.model.N, within=pyo.Binary, default=0)
        # rejection rate
        self.model.Ml = pyo.Param(within=pyo.NonNegativeIntegers)
        self.model.L = pyo.RangeSet(0, self.model.Ml)
        self.model.rejection_rate = pyo.Param(self.model.L, within=pyo.NonNegativeReals, default=0)
        ###########################################################################
        # Problem variables
        ###########################################################################
        # number of enqueued requests
        self.model.x = pyo.Var(self.model.N, self.model.F, domain=PYO_VAR_TYPE)
        # number of forwarded requests
        self.model.y = pyo.Var(self.model.N, self.model.N, self.model.F, domain=PYO_VAR_TYPE)
        # number of rejected requests
        self.model.z = pyo.Var(self.model.N, self.model.F, domain=PYO_VAR_TYPE)
        self.model.prl = pyo.Var(self.model.N, self.model.F, self.model.L, domain=pyo.Binary)
        self.model.processing_reject = pyo.Var(self.model.N, self.model.F, domain=pyo.NonNegativeReals)
        ###########################################################################
        # Constraints
        ###########################################################################
        self.model.no_traffic_loss = pyo.Constraint(self.model.N, self.model.F, rule=self.no_traffic_loss)
        self.model.offload_only_to_neighbors = pyo.Constraint(
            self.model.N, self.model.N, self.model.F, rule=self.offload_only_to_neighbors
        )
        self.model.count_processing1 = pyo.Constraint(self.model.N, self.model.F, rule=self.count_processing1)
        self.model.count_processing2 = pyo.Constraint(self.model.N, self.model.F, rule=self.count_processing2)
        self.model.count_processing3 = pyo.Constraint(self.model.N, self.model.F, rule=self.count_processing3)

    @staticmethod
    def no_traffic_loss(model, n, f):
        return model.x[n, f] + model.z[n, f] + sum(model.y[n, m, f] for m in model.N) == model.incoming_load[n, f]

    @staticmethod
    def offload_only_to_neighbors(model, n, m, f):
        return model.y[n, m, f] <= model.incoming_load[n, f] * model.neighborhood[n, m]

    @staticmethod
    def count_processing1(model, n, f):
        return model.x[n, f] + sum(model.y[m, n, f] for m in model.N) == sum(l * model.prl[n, f, l] for l in model.L)

    @staticmethod
    def count_processing2(model, n, f):
        return sum(model.prl[n, f, l] for l in model.L) == 1

    @staticmethod
    def count_processing3(model, n, f):
        return model.processing_reject[n, f] == sum(model.rejection_rate[l] * model.prl[n, f, l] for l in model.L)


class SimpleCentralizedLMM(BaseLoadManagementModel):
    def __init__(self):
        super().__init__()
        self.name = "SimpleCentralizedLMM"
        ###########################################################################
        # Objective function
        ###########################################################################
        self.model.OBJ = pyo.Objective(rule=self.minimize_rejections)

    @staticmethod
    def minimize_rejections(model):
        return sum(model.z[n, f] for n in model.N for f in model.F) + sum(
            model.processing_reject[n, f] for n in model.N for f in model.F
        )


class WeightedSCLMM(BaseLoadManagementModel):
    def __init__(self):
        super().__init__()
        self.name = "WeightedSCLMM"
        ###########################################################################
        # Objective function
        ###########################################################################
        self.model.OBJ = pyo.Objective(rule=self.minimize_weighted_rejections)

    @staticmethod
    def minimize_weighted_rejections(model):
        return (
            0.2 * sum(model.z[n, f] for n in model.N for f in model.F)
            + 0.8 * sum(model.processing_reject[n, f] for n in model.N for f in model.F)
        ) / sum(model.incoming_load[n, f] for n in model.N for f in model.F)
