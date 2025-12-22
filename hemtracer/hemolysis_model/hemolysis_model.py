from __future__ import annotations
from collections import namedtuple
from collections.abc import Callable
from re import I
import numpy as np
from numpy.typing import NDArray
from enum import Enum

from hemtracer.rbc_model.rbc_model import RBCModel


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

PoreFormationCoefficients = namedtuple('PoreFormationCoefficients', ['h', 'k'])
r"""
Named tuple for pore formation coefficients. Used in :class:`PoreFormationCorrelation`.
:param h: Coefficient :math:`h`.
:type h: float
:param k: Coefficient :math:`k`.
:type k: float
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

class PoreFormationCorrelation(PoreFormationCoefficients, Enum):
    r"""
    Represents sets of empirical parameters :math:`(h, k)` for the pore formation model :math:`\frac{dIH}{dt} = h \sigma^k A_p`, where :math:`A_p` is the pore area.
    """
    DING_HUMAN_STRAINBASED = 2.187e-6, 0.927        #: Ding et al. :cite:p:`dingShearInducedHemolysisSpecies2015` data fitted to strain-based pore formation model

class PowerLawModel:
    r"""A power law hemolysis model is a model that, given a scalar measure for fluid stress (in our case a representative shear rate), predicts the hemolysis index along a pathline. This is done by integrating an experimental correlation for the hemolysis index along the pathline. For details, see :ref:`hemolysis-models`.
    """

    def __init__(self, scalar_shear: RBCModel | str, hemolysis_correlation: IHCorrelation = IHCorrelation.GIERSIEPEN, mu: float | str = 3.5e-3, integration_scheme: str = 'basic') -> None:
        """
        Initialization defines all parameters to use. They cannot be changed afterwards.

        :param scalar_shear: Model to compute scalar shear rate, or name of attribute containing representative shear rate, e.g., 'Geff' if available from an Eulerian simulation.
        :type scalar_shear: RBCModel | str
        :param hemolysis_correlation: Hemolysis correlation to use.
        :type hemolysis_correlation: HemolysisCorrelation
        :param mu: Dynamic viscosity of blood. Defaults to 3.5e-3. If a string is given, it is assumed to be the name of an attribute containing the (local) dynamic viscosity.
        :type mu: float | str
        :param integration_scheme: Integration scheme for hemolysis correlation. Integration schemes defined according to Taskin et al. :cite:p:`taskinEvaluationEulerianLagrangian2012`. Valid options are 'basic' (HI1), 'timeDiff' (HI2), 'linearized' (HI3), 'mechDose' (HI4), 'effTime' (HI5). Defaults to 'basic'.
        :type integration_scheme: str
        """

        if isinstance(scalar_shear, RBCModel):
            self._scalar_shear_name = scalar_shear.get_attribute_name()
        else:
            self._scalar_shear_name = scalar_shear

        self._corr_name = hemolysis_correlation.name
        self._integration_scheme_name = integration_scheme

        r"""
        Obtain coefficients for definitions in accordance with Taskin et al.
        Coefficient C for hemolysis correlation :math:`HI = C t^\alpha \sigma^\beta`. This corresponds to the definition in Taskin et al. :cite:p:`taskinEvaluationEulerianLagrangian2012`. The internal values of this class are thus different from the definitions in :class:`HemolysisCorrelation`, i.e., :math:`C = A_\mathrm{Hb}, \, \alpha = \beta_\mathrm{Hb}, \, \beta = \alpha_\mathrm{Hb}`. This is to allow for direct comparison of the numerical schemes with the formulations in Taskin et al. :cite:p:`taskinEvaluationEulerianLagrangian2012`.
        """
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
    
    def compute_hemolysis(self, t: NDArray, G: NDArray, mu: NDArray) -> NDArray:
        """
        Compute hemolysis along pathline. Called by :class:`HemolysisSolver`.

        :param t: Time steps.
        :type t: NDArray
        :param G: Scalar shear rate.
        :type G: NDArray
        :param mu: Dynamic viscosity.
        :type mu: NDArray
        :return: Hemolysis index.
        :rtype: NDArray
        """

        return self._compute_IH(t, np.squeeze(G)*np.squeeze(mu))
    
    def _compute_HI1(self, t: NDArray, tau: NDArray) -> NDArray:
        """
        Computes index of hemolysis by directly integrating experimental correlation.

        :param t: Time steps.
        :type t: NDArray
        :param tau: Scalar representative stress.
        :type tau: NDArray
        :return: Hemolysis index.
        :rtype: NDArray
        """

        dt = np.diff(t)

        IH = np.zeros_like(t)
        IH[1:] = np.cumsum(self._C * dt**self._alpha * tau[1:]**self._beta)

        return IH
    
    def _compute_HI2(self, t: NDArray, tau: NDArray) -> NDArray:
        """
        Computes index of hemolysis by incorporating time derivative in experimental correlation.

        :param t: Time steps.
        :type t: NDArray
        :param tau: Scalar representative stress.
        :type tau: NDArray
        :return: Hemolysis index.
        :rtype: NDArray
        """

        IH = np.zeros_like(t)

        dt = np.diff(t)
        t_power = t[1:]**(self._alpha-1)
        IH[1:] = self._C * self._alpha * np.cumsum(t_power * tau[1:]**self._beta * dt)

        return IH

    def _compute_HI3(self, t: NDArray, tau: NDArray) -> NDArray:
        """
        Computes index of hemolysis by summing linearized damage.

        :param t: Time steps.
        :type t: NDArray
        :param tau: Scalar representative stress.
        :type tau: NDArray
        :return: Hemolysis index.
        :rtype: NDArray
        """

        IH = np.zeros_like(t)

        dt = np.diff(t)
        partial_sum = np.cumsum(dt * tau[1:]**(self._beta/self._alpha))
        IH[1:] = self._C * partial_sum**self._alpha

        return IH
    
    def _compute_HI4(self, t: NDArray, tau: NDArray) -> NDArray:
        """
        Computes index of hemolysis by accumulating mechanical dose (Grigioni et al. :cite:p:`grigioniNovelFormulationBlood2005`).

        :param t: Time steps.
        :type t: NDArray
        :param tau: Scalar representative stress.
        :type tau: NDArray
        :return: Hemolysis index.
        :rtype: NDArray
        """

        D0 = 0      # Initial dose (can be defined differently to account for damage accumulation)

        dt = np.diff(t)
        partial_sum = np.cumsum(dt * tau[1:]**(self._beta/self._alpha)) + D0
        IH = np.zeros_like(t)
        IH[1:] = self._alpha * self._C * partial_sum**(self._alpha-1) * tau[1:]**(self._beta/self._alpha) * dt

        return IH
    
    def _compute_HI5(self, t: NDArray, tau: NDArray) -> NDArray:
        """
        Computes index of hemolysis by using virtual time step approach (Goubergrits and Affeld :cite:p:`goubergritsNumericalEstimationBlood2004`).

        :param t: Time steps.
        :type t: NDArray
        :param tau: Scalar representative stress.
        :type tau: NDArray
        :return: Hemolysis index.
        :rtype: NDArray
        """

        IH = np.zeros_like(t)
        dt = np.diff(t)
        C_tau_beta = self._C * tau[1:]**self._beta

        for i in range(1, len(t)-1):
            t_eff = (IH[i-1] / C_tau_beta[i-1])**(1/self._alpha)
            IH[i] = C_tau_beta[i] * (t_eff + dt[i-1])**self._alpha
        IH[-1] = IH[-2]

        return IH
    
    def get_name(self) -> str:
        """
        Get the name of the power law hemolysis model.

        :return: The name of the power law hemolysis model.
        :rtype: str
        """

        return self._corr_name + '_' + self._integration_scheme_name
    
    def get_attribute_name(self) -> str:
        """
        Get the name of the attribute that will be added to pathlines.

        :return: The name of the attribute that will be added to pathlines.
        :rtype: str
        """

        return 'IH_' + self.get_name() + '_' + self._scalar_shear_name
    
    def get_scalar_shear_name(self) -> str:
        """
        Get the name of the scalar shear rate attribute.

        :return: The name of the scalar shear rate attribute.
        :rtype: str
        """

        return self._scalar_shear_name

class PoreFormationModel:
    """Calculates hemolysis based on pore formation mechanism.
    """
    def __init__(self, effective_shear: RBCModel | str, fluid_shear: RBCModel | str,
                 hemolysis_correlation: PoreFormationCorrelation = PoreFormationCorrelation.DING_HUMAN_STRAINBASED, mu: float | str = 3.5e-3) -> None:
        """
        Initialization defines all parameters to use. They cannot be changed afterwards.

        :param effective_shear: Model to compute effective shear rate, or name of attribute containing effective shear rate, e.g., 'Geff' if available from an Eulerian simulation.
        :type effective_shear: RBCModel | str
        :param fluid_shear: Model to compute fluid shear rate, or name of attribute containing fluid shear rate, e.g., 'Gfluid' if available from an Eulerian simulation.
        :type fluid_shear: RBCModel | str
        :param hemolysis_correlation: Hemolysis correlation to use.
        :type hemolysis_correlation: PoreFormationCorrelation
        :param mu: Dynamic viscosity of blood. Defaults to 3.5e-3. If a string is given, it is assumed to be the name of an attribute containing the (local) dynamic viscosity.
        :type mu: float | str
        """

        if isinstance(effective_shear, RBCModel):
            self._effective_shear_name = effective_shear.get_attribute_name()
        else:
            self._effective_shear_name = effective_shear

        if isinstance(fluid_shear, RBCModel):
            self._fluid_shear_name = fluid_shear.get_attribute_name()
        else:
            self._fluid_shear_name = fluid_shear

        self._corr_name = hemolysis_correlation.name

        self._h = hemolysis_correlation.h
        self._k = hemolysis_correlation.k
        self._mu = mu

    def get_name(self) -> str:
        """
        Get the name of the pore formation hemolysis model.
        :return: The name of the pore formation hemolysis model.
        :rtype: str
        """

        return self._corr_name + '_PoreFormation'
    
    def get_attribute_name(self) -> str:
        """
        Get the name of the attribute that will be added to pathlines.

        :return: The name of the attribute that will be added to pathlines.
        :rtype: str
        """

        return 'IH_' + self.get_name() + '_' + self._effective_shear_name + '_' + self._fluid_shear_name
    
    def get_effective_shear_name(self) -> str:
        """
        Get the name of the effective shear rate attribute.

        :return: The name of the effective shear rate attribute.
        :rtype: str
        """

        return self._effective_shear_name
    
    def get_fluid_shear_name(self) -> str:
        """
        Get the name of the fluid shear rate attribute.

        :return: The name of the fluid shear rate attribute.
        :rtype: str
        """

        return self._fluid_shear_name
    
    def compute_hemolysis(self, t: NDArray, G_eff: NDArray, tau_fluid: NDArray) -> NDArray:
        """
        Compute hemolysis along pathline. Called by :class:`HemolysisSolver`.

        :param t: Time steps.
        :type t: NDArray
        :param G_eff: Effective scalar shear rate.
        :type G_eff: NDArray
        :param G_fluid: Fluid scalar shear rate.
        :type G_fluid: NDArray
        :return: Hemolysis index.
        :rtype: NDArray
        """

        dt = np.diff(t)

        IH = np.zeros_like(t)
        Ap = self._calcPoreArea(G_eff[:-1])
        IH[1:] = np.cumsum(self._h * tau_fluid[:-1]**self._k * Ap * dt)

        return IH
    
    def _calcPoreArea(self, G_eff: NDArray) -> NDArray:
        """
        Calculate pore area based on effective shear rate.

        :param G_eff: Effective scalar shear rate.
        :type G_eff: NDArray
        :return: Pore area.
        :rtype: NDArray
        """

        p = [ -13.41, 37.31, -48.91, 32.39, 0.63, -0.16 ]
        G1 = 3750
        G2 = 42000

        A_p = np.where(G_eff < G1, np.zeros_like(G_eff), 
              np.where(G_eff > G2, np.polyval(p, np.ones_like(G_eff)),
                                   np.polyval(p, G_eff/G2)))
        
        return A_p