"""This module defines cell models used in hemolysis computations, i.e., stress-based and strain-based models for cell deformation."""
from __future__ import annotations
import numpy as np
from collections.abc import Callable
from typing import Tuple, Dict
from numpy.typing import NDArray
from hemtracer._definitions import Vector3, Vector9, Vector12, Tensor3

class RBCModel:
    """Abstract base class for any model that computes a scalar shear rate along a pathline. These models assume a certain red blood cell (RBC) behavior, hence the name RBC Model. They can be categorized into two groups: stress-based and strain-based models. Stress-based models assume direct deformation of the cell in response to stress, reducing the instantaneous three-dimensional local fluid strain to a representative scalar shear rate. Strain-based models take into account the characteristic membrane relaxation time, employing a differential equation to explicitly resolve cell deformation in response to the acting flow forces. They then compute a scalar shear rate (so-called effective shear) from cell deformation.
    """

    def set_time_dependent_quantitites(self, t0: float, tend: float, 
                                       time_points: NDArray | None = None,
                                       dv: Callable[[float],Vector9] | None = None, 
                                       omega: Callable[[float],Vector3] | None = None, 
                                       x: Callable[[float], Vector3] | None = None,
                                       v: Callable[[float], Vector3] | None = None,
                                       init: Dict[str, float] | None = None) -> None:
        """
        Set time-dependent quantities for scalar shear rate model and check if all required quantities are there. Start and end time as well as velocity gradient are always required. If omega is not defined, it is set to zero, assuming a stationary frame of reference. The tank-treading model with pathline additionally requires x and v. This is handled by :class:`hemtracer.hemolysis_solver.HemolysisSolver`.

        :meta private:
        :param t0: Start time.
        :type t0: float
        :param tend: End time.
        :type tend: float
        :param time_points: Time points at which to sample velocity gradients. 
        :type time_points: NDArray
        :param dv: Velocity gradient tensor (in vector form) as a function of pathline time.
        :type dv: Callable[[float],Vector9]
        :param omega: MRF angular velocity vector as a function of pathline time.
        :type omega: Callable[[float],Vector3]
        :param x: Relevant position measure as a function of pathline time, e.g., absolute coordinates or distance from rotational center. Only required for tank-treading with pathline correction.
        :type x: Callable[[float],Vector3]
        :param v: Velocity as a function of pathline time. Only required for tank-treading with pathline correction.
        :type v: Callable[[float],Vector3]
        :param init: Dict with initial values for attributes on pathline, only required for specific shear initial condition.
        :type init: Dict[str, float]
        """

        """Sanity checks."""
        # Start and end time are always required.
        if t0 is None or tend is None:
            raise ValueError('Start and end time have to be defined.')
        
        # Check validity of start and end time.
        if t0 >= tend:
            raise ValueError('Start time has to be smaller than end time.')

        # Velocity gradient is always required.
        if dv is None:
            raise ValueError('Velocity gradient not defined.')
        
        # If omega is not defined, set it to zero, assuming stationary frame of reference.
        if omega is None:
            omega = lambda t: np.zeros(3)

        """Assign quantities."""
        self._t0 = t0 # Start time.
        self._tend = tend # End time.
        self._time_points = time_points # Time points at which to sample velocity gradients.
        self._dv = dv # Velocity gradient tensor.
        self._omega = omega # MRF angular velocity vector.
        self._init = init # Initial values for attributes on pathline.

    def _compute_strain_tensor(self, t: float) -> Tensor3:
        """
        Compute strain tensor at time t.

        :param t: Time.
        :type t: float
        :return: Strain tensor.
        :rtype: Tensor3
        """

        dv_i = self._dv(t).reshape(3, 3).T
        E = 0.5 * (dv_i + dv_i.T)

        return E
    
    def _compute_second_invariant(self, T: Tensor3) -> float:
        """
        Compute second invariant of tensor.

        :param T: Tensor.
        :type T: Tensor3
        :return: Second invariant.
        :rtype: float
        """

        return 0.5*(np.trace(T)**2 - np.trace(np.matmul(T, T)))
    
    def _compute_shear_rate_second_invariant(self, E: Tensor3) -> float:
        """
        Compute shear rate from strain tensor using second invariant.

        :param E: Strain tensor.
        :type E: Tensor3
        :return: Shear rate.
        :rtype: float
        """

        G = 2*np.sqrt(np.abs(self._compute_second_invariant(E)))

        return G

    def get_name(self) -> str:
        """
        Get name of blood damage model. Should not contain spaces. Has to be implemented in subclasses.

        :return: Name of blood damage model.
        :rtype: str
        """

        raise NotImplementedError('get_name has to be implemented in subclasses.')
    
    def get_attribute_name(self) -> str:
        """
        Get name of resulting scalar shear rate attribute on pathlines.

        :return: Name of scalar shear rate attribute.
        :rtype: str
        """

        return 'G_' + self.get_name()
    
    def compute_representative_shear(self) -> Tuple[NDArray, NDArray, NDArray|None]:
        """
        Solve blood damage model to compute scalar shear rate and return times and scalar shear rate along pathline. 
        The third return can be used to return the solution of ODE models.

        :return: Tuple of times and scalar shear rate along pathline.
        :rtype: Tuple[NDArray, NDArray, NDArray|None]
        """

        raise NotImplementedError('compute_representative_shear has to be implemented in subclasses.')