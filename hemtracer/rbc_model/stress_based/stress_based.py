from typing import Tuple
from numpy.typing import NDArray
from hemtracer._definitions import Tensor3
from hemtracer.rbc_model import RBCModel
import numpy as np

class StressBasedModel(RBCModel):
    r"""
    Represents a stress-based blood damage model. These models reduce the instantaneous three-dimensional local fluid strain :math:`\mathbf{E}` to a representative scalar shear rate :math:`G_s`. They assume that RBCs deform instantly in response to changes in fluid stress. The local velocity gradient tensor is taken from pathlines or sampled at fixed time intervals and the representative scalar shear rate is computed from the strain tensor.
    """

    def __init__(self) -> None:
        """
        Initialize stress-based model. By default, use same time points as pathline, i.e., do not sample.
        """

        self._sampling_rate = None

    def set_sampling_rate(self, sampling_rate: float) -> None:
        """
        Set sampling rate of velocity gradients along pathline for stress-based model.
        If sampling rate is positive, it is interpreted as the time interval between two consecutive samples. If sampling rate is negative, it is interpreted as the number of samples per time interval.

        :param sampling_rate: Sampling rate.
        :type sampling_rate: float
        """

        self._sampling_rate = sampling_rate
    
    def compute_representative_shear(self) -> Tuple[NDArray, NDArray]:
        """
        Compute representative scalar shear rate from flow field. Called by :class:`HemolysisSolver`.

        :meta private:
        :return: Tuple containing time steps and representative scalar shear rate.
        :rtype: Tuple[NDArray, NDArray]
        """

        if self._sampling_rate is not None:
            if self._sampling_rate > 0:
                t = np.arange(self._t0, self._tend, self._sampling_rate)
            else: # subsample given time points
                if self._time_points is None:
                    raise AttributeError('No time points defined.')
                n = abs(int(self._sampling_rate)) # number of time points in each interval
                t = []
                for i in range(1, len(self._time_points)):
                    t.extend(np.linspace(self._time_points[i-1], self._time_points[i], n, endpoint=False))
                t = np.asarray(t)
        else:
            if self._time_points is None:
                raise AttributeError('No time points defined.')
            t = self._time_points

        G = np.asarray([self._compute_representative_shear(self._compute_strain_tensor(ti)) for ti in t])

        return (t, G)

    def _compute_representative_shear(self, E: Tensor3) -> float:
        """
        Reduce 3D strain tensor to representative scalar shear rate.

        :param E: 3D strain tensor.
        :type E: Tensor3
        :return: Scalar shear rate.
        :rtype: float
        """

        raise NotImplementedError('_compute_representative_shear has to be implemented in subclasses.')
    
class Bludszuweit(StressBasedModel):
    r"""
    Represents the stress-based :ref:`bludszuweit-model`.
    """

    def get_name(self) -> str:
        """
        Get name of blood damage model.

        :return: Name of the blood damage model.
        :rtype: str
        """

        return 'stress-bludszuweit'

    def _compute_representative_shear(self, E: Tensor3) -> float:
        """
        Reduce 3D strain tensor to representative scalar shear rate using the law proposed by Bludszuweit (1995)

        :param E: 3D strain tensor.
        :type E: Tensor3
        :return: Scalar shear rate.
        :rtype: float
        """

        t_xy = E[0,1]
        t_xz = E[0,2]
        t_yz = E[1,2]
        s_xx = E[0,0]
        s_yy = E[1,1]
        s_zz = E[2,2]

        G = 2/np.sqrt(3) * np.sqrt(
                (s_xx**2   + s_yy**2   + s_zz**2  ) 
            -   (s_xx*s_yy + s_xx*s_zz + s_yy*s_zz)
            + 3*(t_xy**2   + t_xz**2   + t_yz**2  ))
        return G
    
class FaghihSharp(StressBasedModel):
    r"""
    Represents the stress-based :ref:`faghih-sharp-model` that weighs extensional and shear stresses differently.
    """

    def get_name(self) -> str:
        """
        Get name of blood damage model.

        :return: Name of the blood damage model.
        :rtype: str
        """

        return 'stress-faghih-sharp'
    
    def _compute_representative_shear(self, E: Tensor3) -> float:
        """
        Reduce 3D strain tensor to representative scalar shear rate using the law proposed by Faghih and Sharp.
        """

        C_n = 33.79/np.sqrt(3)

        t_xy = E[0,1]
        t_xz = E[0,2]
        t_yz = E[1,2]
        s_xx = E[0,0]
        s_yy = E[1,1]
        s_zz = E[2,2]

        G = 2*np.sqrt( C_n**2 * (
                       s_xx**2   + s_yy**2   + s_zz**2
                    - (s_xx*s_yy + s_xx*s_zz + s_yy*s_zz))
                    + (t_xy**2   + t_xz**2   + t_yz**2) )
        return G


class Frobenius(StressBasedModel):
    r"""
    Computes a representative scalar from instantaneous fluid strain using the :ref:`frobenius-model`.
    """

    def get_name(self) -> str:
        """
        Get name of blood damage model.

        :return: Name of the blood damage model.
        :rtype: str
        """

        return 'stress-frobenius'

    def _compute_representative_shear(self, E: Tensor3) -> float:
        """
        Reduce 3D strain tensor to representative scalar shear rate using Frobenius norm on strain tensor.

        :param E: 3D strain tensor.
        :type E: Tensor3
        :return: Scalar shear rate.
        :rtype: float
        """

        G = np.sqrt(2)*np.linalg.norm(E, ord='fro')
        return float(G)

class SecondInvariant(StressBasedModel):
    r"""
    Computes a representative scalar from instantaneous fluid strain using the :ref:`second-invariant-model`.
    """

    def get_name(self) -> str:
        """
        Get name of blood damage model.

        :return: Name of the blood damage model.
        :rtype: str
        """

        return 'stress-second-invariant'

    def _compute_representative_shear(self, E: Tensor3) -> float:
        """
        Reduce 3D strain tensor to representative scalar shear rate using second strain invariant.

        :param E: 3D strain tensor.
        :type E: Tensor3
        :return: Scalar shear rate.
        :rtype: float
        """

        return self._compute_shear_rate_second_invariant(E)