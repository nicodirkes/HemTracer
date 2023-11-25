from __future__ import annotations
from collections import namedtuple
from collections.abc import Callable
import numpy as np
from numpy.typing import ArrayLike, NDArray
from enum import Enum


CorrelationCoefficients = namedtuple('CorrelationCoefficients', ['A_Hb', 'alpha_Hb', 'beta_Hb'])
r"""
Named tuple for correlation coefficients. Used in :class:`IHCorrelation`.

:param A_Hb: Coefficient :math:`A_\mathrm{Hb}`.
:type A_Hb: float
:param alpha_Hb: Coefficient :math:`\alpha_\mathrm{Hb}`.
:type alpha_Hb: float
:param beta_Hb: Coefficient :math:`\beta_\mathrm{Hb}`.
:type beta_Hb: float
"""

class IHCorrelation(CorrelationCoefficients, Enum):
    r"""
    Represents sets of empirical parameters :math:`(A_\mathrm{Hb}, \alpha_\mathrm{Hb}, \beta_\mathrm{Hb})` for the correlation :math:`IH = A_\mathrm{Hb} \sigma^{\alpha_\mathrm{Hb}} t^{\beta_\mathrm{Hb}}`. Magnitude of :math:`A_{Hb}` is chosen such that :math:`IH` is in percent.
    """
    
    GIERSIEPEN = 3.62e-5, 2.416, 0.785              #: Giersiepen et al. :cite:p:`giersiepenEstimationShearStressRelated1990`
    SONG = 1.8e-6, 1.991, 0.765                     #: Song et al. :cite:p:`songComputationalFluidDynamics2003`
    ZHANG = 1.228e-5, 1.9918, 0.6606                #: Zhang et al. :cite:p:`zhangStudyFlowInducedHemolysis2011`
    DING_HUMAN = 3.458e-6, 2.0639, 0.2777           #: Ding et al. :cite:p:`dingShearInducedHemolysisSpecies2015`
    DING_PORCINE = 6.701e-4, 1.0981, 0.2778         #: Ding et al. :cite:p:`dingShearInducedHemolysisSpecies2015`
    DING_BOVINE = 9.772e-5, 1.4445, 0.2076          #: Ding et al. :cite:p:`dingShearInducedHemolysisSpecies2015`
    GESENHUES_OPTIMIZED = 2.3212e-4, 1.4949, 0.33   #: Gesenhues et al. :cite:p:`gesenhuesStrainBasedBloodDamage2016`

class PowerLawModel:
    r"""A power law hemolysis model is a model that, given a scalar measure for fluid stress (in our case a representative shear rate), predicts the hemolysis index along a pathline. This is done by integrating an experimental correlation for the hemolysis index along the pathline. For details, see :ref:`hemolysis-models`.
    """

    _hemolysis_correlation: IHCorrelation = None  
    """
    Power law coefficients to use.
    """

    _corr_name: str = None  
    """
    Name of correlation used.
    """

    _mu: float = None  
    """
    Dynamic viscosity of blood.
    """

    _C: float = None  
    r"""
    Coefficient C for hemolysis correlation :math:`HI = C t^\alpha \sigma^\beta`. This corresponds to the definition in Taskin et al. :cite:p:`taskinEvaluationEulerianLagrangian2012`. The internal values of this class are thus different from the definitions in :class:`HemolysisCorrelation`, i.e., :math:`C = A_\mathrm{Hb}, \, \alpha = \beta_\mathrm{Hb}, \, \beta = \alpha_\mathrm{Hb}`. This is to allow for direct comparison of the numerical schemes with the formulations in Taskin et al. :cite:p:`taskinEvaluationEulerianLagrangian2012`.
    """

    _alpha: float = None  
    """
    Coefficient alpha in hemolysis correlation.
    """

    _beta: float = None  
    """
    Coefficient beta in hemolysis correlation.
    """

    _integration_scheme_name: str = None  
    """
    Name of scheme used to numerically integrate hemolysis correlation.
    """

    _compute_IH: Callable[[ArrayLike, ArrayLike], ArrayLike] = None  
    """
    Function that computes hemolysis index from time and scalar shear rate using the desired integration scheme.
    """

    def __init__(self, hemolysis_correlation: IHCorrelation = IHCorrelation.GIERSIEPEN, mu: float = 3.5e-3, integration_scheme: str = 'basic') -> None:
        """
        Initialization defines all parameters to use. They cannot be changed afterwards.

        :param hemolysis_correlation: Hemolysis correlation to use.
        :type hemolysis_correlation: HemolysisCorrelation
        :param mu: Dynamic viscosity of blood. Defaults to 3.5e-3.
        :type mu: float
        :param integration_scheme: Integration scheme for hemolysis correlation. Integration schemes defined according to Taskin et al. :cite:p:`taskinEvaluationEulerianLagrangian2012`. Valid options are 'basic' (HI1), 'timeDiff' (HI2), 'linearized' (HI3), 'mechDose' (HI4), 'effTime' (HI5). Defaults to 'basic'.
        :type integration_scheme: str
        """

        self._corr_name = hemolysis_correlation.name
        self._integration_scheme_name = integration_scheme

        """Obtain coefficients for definitions in accordance with Taskin et al."""
        self._C = hemolysis_correlation.A_Hb
        self._alpha = hemolysis_correlation.beta_Hb
        self._beta = hemolysis_correlation.alpha_Hb
        self._mu = mu

        """Define integration scheme."""
        match integration_scheme:
            case 'basic':
                self._compute_IH = self._compute_HI1
            case 'timeDiff':
                self._compute_IH = self._compute_HI2
            case 'linearized':
                self._compute_IH = self._compute_HI3
            case 'mechDose':
                self._compute_IH = self._compute_HI4
            case 'effTime':
                self._compute_IH = self._compute_HI5
            case _:
                raise ValueError('Unknown integration scheme.')
    
    def compute_hemolysis(self, t: ArrayLike, G: ArrayLike) -> NDArray:
        """
        Compute hemolysis along pathline. Called by :class:`HemolysisSolver`.

        :param t: Time steps.
        :type t: ArrayLike
        :param G: Scalar shear rate.
        :type G: ArrayLike
        :return: Hemolysis index.
        :rtype: NDArray
        """

        return self._compute_IH(t, G)
    
    def _compute_HI1(self, t: ArrayLike, G: ArrayLike) -> NDArray:
        """
        Computes index of hemolysis by directly integrating experimental correlation.

        :param t: Time steps.
        :type t: ArrayLike
        :param G: Scalar shear rate.
        :type G: ArrayLike
        :return: Hemolysis index.
        :rtype: NDArray
        """

        IH = np.zeros_like(t)

        for i in range(1, len(t)):
            tau = self._mu * G[i-1]
            dt = t[i] - t[i-1]
            IH[i] = IH[i-1] + self._C * dt**self._alpha * tau ** self._beta

        return IH
    
    def _compute_HI2(self, t: ArrayLike, G: ArrayLike) -> NDArray:
        """
        Computes index of hemolysis by incorporating time derivative in experimental correlation.

        :param t: Time steps.
        :type t: ArrayLike
        :param G: Scalar shear rate.
        :type G: ArrayLike
        :return: Hemolysis index.
        :rtype: NDArray
        """

        IH = np.zeros_like(t)

        for i in range(1, len(t)):
            tau = self._mu * G[i-1]
            dt = t[i] - t[i-1]
            IH[i] = IH[i-1] + self._C * t[i]**(self._alpha-1) * tau**self._beta * dt

        return IH

    def _compute_HI3(self, t: ArrayLike, G: ArrayLike) -> NDArray:
        """
        Computes index of hemolysis by summing linearized damage.

        :param t: Time steps.
        :type t: ArrayLike
        :param G: Scalar shear rate.
        :type G: ArrayLike
        :return: Hemolysis index.
        :rtype: NDArray
        """

        IH = np.zeros_like(t)
        partial_sum = 0

        for i in range(1, len(t)):
            tau = self._mu * G[i-1]
            dt = t[i] - t[i-1]
            partial_sum += dt * tau**(self._beta/self._alpha)
            IH[i] = self._C * partial_sum**self._alpha

        return IH
    
    def _compute_HI4(self, t: ArrayLike, G: ArrayLike) -> NDArray:
        """
        Computes index of hemolysis by accumulating mechanical dose (Grigioni et al. :cite:p:`grigioniNovelFormulationBlood2005`).

        :param t: Time steps.
        :type t: ArrayLike
        :param G: Scalar shear rate.
        :type G: ArrayLike
        :return: Hemolysis index.
        :rtype: NDArray
        """

        IH = np.zeros_like(t)
        D0 = 0      # Initial dose (can be defined differently to account for damage accumulation)
        partial_sum = D0

        for i in range(1, len(t)):
            tau = self._mu * G[i-1]
            dt = t[i] - t[i-1]
            partial_sum += dt * tau**(self._beta/self._alpha)
            IH[i] = IH[i-1] + self._alpha * self._C * partial_sum**(self._alpha-1) * tau**(self._beta/self._alpha) * dt

        return IH
    
    def _compute_HI5(self, t: ArrayLike, G: ArrayLike) -> NDArray:
        """
        Computes index of hemolysis by using virtual time step approach (Goubergrits and Affeld :cite:p:`goubergritsNumericalEstimationBlood2004`).

        :param t: Time steps.
        :type t: ArrayLike
        :param G: Scalar shear rate.
        :type G: ArrayLike
        :return: Hemolysis index.
        :rtype: NDArray
        """

        IH = np.zeros_like(t)

        for i in range(1, len(t)):
            tau = self._mu * G[i]
            dt = t[i] - t[i-1]
            t_eff = (IH[i-1] / (self._C * tau**self._beta))**(1/self._alpha)
            IH[i] = self._C * (t_eff + dt)**self._alpha * tau**self._beta

        return IH
    
    def get_name(self) -> str:
        """
        Get the name of the power law hemolysis model.

        :return: The name of the power law hemolysis model.
        :rtype: str
        """

        return self._corr_name + '_' + self._integration_scheme_name