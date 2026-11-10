import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import netket.experimental as nkx
from netket.jax import HashablePartial
from netket.logging.json_log import JsonLog
from netket.operator import AbstractOperator
from netket.utils import mpi, timing
from netket.utils.dispatch import dispatch
from netket.utils.types import PyTree, Union, Array, Callable
from netket.vqs import VariationalState
from netket.experimental.dynamics import Integrator
from netket.hilbert import AbstractHilbert
from netket.experimental.driver.tdvp_common import TDVPBaseDriver, odefun

from .state import VariationalVectorState, _normalize, _compute_norm
from .utils import to_matrix
from netket.stats import Stats

class ExactDynamics(TDVPBaseDriver):
    """ Does not go to stupid logspace ! """
    def __init__(
        self,
        hilbert: AbstractHilbert,
        operator: AbstractOperator,
        vector_array: Union[Array, VariationalState],
        integrator: Integrator,
        *,
        t0: float = 0.0,
        propagation_type: str = "real",
        error_norm: Union[str, Callable] = "euclidean",
        sparse: bool = True,
    ):        
        if isinstance(vector_array, VariationalVectorState):
            variational_state = vector_array
        elif isinstance(vector_array, nk.vqs.VariationalState):
            variational_state = VariationalVectorState(hilbert, vector_array.to_array(normalize=False), normalize=False)
        else:
            vector_array = jnp.array(vector_array, dtype=jnp.complex128)
            variational_state = VariationalVectorState(hilbert, vector_array, normalize=False)
        
        self.propagation_type = propagation_type
        if propagation_type == "real":
            self._dynamics_factor = 1j
            self._subtract_mean = False
            variational_state.to_complex()
        elif propagation_type == "imag":
            self._dynamics_factor = 1
            self._subtract_mean = True
        else:
            raise ValueError("propagation_type must be one of 'real', 'imag'")

        self._operator_matrix = None
        self._sparse = sparse
        self._time_independent_op = isinstance(operator, AbstractOperator)
        
        super().__init__(operator, variational_state, integrator, t0=t0, error_norm=error_norm)

        if self._time_independent_op:
            self.operator_matrix(0.0)
            
    def operator_matrix(self, t):
        if self._time_independent_op:
            if self._operator_matrix is None:
                op = self._generator(t)
                # we cache it here instead
                self._operator_matrix = to_matrix(op, sparse=self._sparse, cache=False)
            return self._operator_matrix
        else:
            op = self._generator(t)
            return to_matrix(op, sparse=self._sparse, cache=False)

@odefun.dispatch
def odefun_tdvp(  # noqa: F811
    state: VariationalVectorState, driver: ExactDynamics, t, w, *, stage=0
):
    # pylint: disable=protected-access
    # the output of this is now parameters = a full Psi
    #
    state.parameters = w
    state.reset()

    H_t = driver.operator_matrix(t)

    # imag:
    # dPsi/dt =  - factor (H-E) Psi = f(t)
    # real:
    # dPsi/dt =  - factor H Psi = f(t)
    HPsi, E = _Opsi_and_expO(H_t, state.vector)
    driver._loss_stats = Stats(mean=E, error_of_mean=0.0)
    dPsi_dt = _schrodinger(HPsi, E, state.vector, driver._dynamics_factor, subtract_mean=driver._subtract_mean)  
    # If parameters are real, then take only real part of the gradient (if it's complex)
    driver._dw = {
        "vector" : dPsi_dt
    }

    if stage == 0:
        # save the info at the initial point
        driver._ode_data = {
            "loss_stats" : driver._loss_stats,
        }

    return driver._dw

def _schrodinger(Hpsi, E, psi, factor, *, subtract_mean):
    if subtract_mean: # imaginary time
        return - factor * (Hpsi - E*psi)
    else:
        return - factor * Hpsi # this is the correct one, you'll get different results

def _Opsi_and_expO(O, v):
    Opsi = O @ v
    expval = (v.conj() @ Opsi).sum()/np.linalg.norm(v)**2
    return Opsi, expval