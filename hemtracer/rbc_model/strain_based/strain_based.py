from hemtracer.rbc_model import RBCModel
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from hemtracer._definitions import Vector3, Vector9, Vector12, Tensor3
from collections.abc import Callable
from collections import namedtuple
from enum import Enum
import numpy as np
from typing import Tuple

MorphologyCoefficients = namedtuple('MorphologyCoefficients', ['f1', 'f2', 'f3', 'f2t', 'f3t'])

class MorphologyModelCoefficients(MorphologyCoefficients, Enum):
    r"""Structure that contains morphology model coefficients :math:`(f_1, f_2, f_3, \tilde{f}_2, \tilde{f}_3)`. The coefficients are defined as follows:

    :math:`f_1` : Recovery coefficient.

    :math:`f_2` : Strain coefficient.

    :math:`f_3` : Vorticity coefficient.

    :math:`\tilde{f}_2`: Coefficient for rotational part of strain tensor (only relevant for :class:`AroraFullEig`).

    :math:`\tilde{f}_3`: Coefficient for rotational part of vorticity tensor (only relevant for :class:`AroraFullEig`).
    """

    ARORA = 5.0, 4.2298e-4, 4.2298e-4, 1.0, 1.0 #: Arora et al. (2004)

class StrainBasedModel(RBCModel):
    """Represents a strain-based blood damage model. These models employ a differential equation to explicitly resolve cell deformation in response to the acting flow forces. In the current implementation, this is constructed in particular for the Arora model (Arora et al. 2004) and all models derived from it, i.e., the simplified Eulerian model :cite:t:`pauli_transient_2013`, the full Eulerian morphology model (Dirkes et al. 2023) and the tank-treading model (Dirkes et al. 2023). In theory, however, also other strain-based models could be implemented as sub-classes. Since an ODE solver is required, these models offer additional configuration options for the solver and the initial condition.
    """

    _initial_condition: NDArray
    """
    Initial condition for solution variable, i.e., shape of cells at the start of pathline integration.
    """

    _method: str
    """
    Method for ODE solver.
    """

    _atol: float
    """
    Absolute tolerance for ODE solver.
    """

    _rtol: float
    """
    Relative tolerance for ODE solver.
    """

    _first_step: float
    """
    Initial step size for ODE solver.
    """

    _max_step: float
    """
    Maximum step size for ODE solver.
    """

    _coeffs: MorphologyModelCoefficients = MorphologyModelCoefficients.ARORA
    """
    Coefficients for morphology model.
    """

    def __init__(self) -> None:
        """
        Arora coefficients (Arora et al. 2004) are used by default.
        """

        super().__init__()
        self.set_initial_condition()
        self.configure_ODE_solver()
    
    def set_coefficients(self, coeffs: MorphologyModelCoefficients) -> None:
        """
        Set coefficients to use for morphology model. If not called, the Arora coefficients (Arora et al. 2004) are used by default.

        :param coeffs: Coefficients for morphology model, has to be a member of the :code:`Enum` class :class:`MorphologyModelCoefficients`, e.g., :code:`hemtracer.rbc_model.strain_based.MorphologyModelCoefficients.ARORA`.
        :type coeffs: MorphologyModelCoefficients
        """

        self._coeffs = coeffs

    def set_initial_condition(self, type: str = 'undeformed') -> None:
        """
        Set initial condition for strain-based model.

        :param type: Type of initial condition, i.e., shape of cells at the start of the pathline. Currently supported: undeformed, steadyShear. 'undeformed' represents an undeformed, i.e., perfectly spherical cell. 'steadyShear' represents the steady state for a simple shear flow. The shear rate is computed using the second invariant of the strain tensor at the initial position. Defaults to 'undeformed'.
        :type type: str
        """

        match type:
            case 'undeformed':
                self._initial_condition = self._initial_condition_undeformed()
            case 'steadyShear':
                self._initial_condition = self._initial_condition_steady()
            case _:
                raise ValueError('Initial condition type unknown.')

    def configure_ODE_solver(self, method: str = 'RK45', 
                             atol: float = 1e-6, rtol: float = 1e-5,
                             first_step: float = 0.0001, max_step: float = 0.1) -> None:
        """
        Configure ODE solver for strain-based model. For details on the parameters, see scipy.integrate.solve_ivp

        :param method: Method for ODE solver. Defaults to 'RK45'.
        :type method: str
        :param atol: Absolute tolerance for ODE solver. Defaults to 1e-6.
        :type atol: float
        :param rtol: Relative tolerance for ODE solver. Defaults to 1e-5.
        :type rtol: float
        :param first_step: Initial step size for ODE solver. Defaults to 0.0001.
        :type first_step: float
        :param max_step: Maximum step size for ODE solver. Defaults to 0.1.
        :type max_step: float
        """

        self._method = method
        self._atol = atol
        self._rtol = rtol
        self._first_step = first_step
        self._max_step = max_step

    def compute_representative_shear(self) -> Tuple[NDArray, NDArray]:
        """
        Solve strain-based model, return effective shear rate and chosen time steps.
        Called by :class:`hemtracer.hemolysis_solver.HemolysisSolver`.

        :meta private:
        :return: Time steps and Effective shear rate.
        :rtype: Tuple[NDArray, NDArray]
        """

        result = solve_ivp(self._RHS, (self._t0, self._tend), self._initial_condition, method=self._method, atol=self._atol, rtol=self._rtol, first_step=self._first_step, max_step=self._max_step)
        t = np.transpose(result.t)
        sol = np.transpose(result.y)
        Geff = self._compute_Geff_from_sol(sol)

        return (t, Geff)

    def _initial_condition_undeformed(self) -> NDArray:
        """
        Set initial condition for undeformed cells. This is the default initial condition for all strain-based models. Has to be implemented by subclasses to be usable.

        :return: A vector of values describing an undeformed cell in the model.
        :rtype: NDArray
        :raises NotImplementedError: If undeformed initial condition is not implemented in this cell model.
        """

        raise NotImplementedError('Undeformed initial condition not implemented in this cell model.')
    
    def _initial_condition_steady(self) -> NDArray:
        """
        Set initial condition for steady shear. Has to be implemented by subclasses to be usable.

        :return: A vector of values describing a cell in steady shear in the model.
        :rtype: NDArray
        :raises NotImplementedError: If steady shear initial condition is not implemented in this cell model.
        """

        raise NotImplementedError('Steady shear initial condition not implemented in this cell model.')
    
    def _compute_Geff_from_sol(self, sol: NDArray) -> NDArray:
        """
        Compute effective shear rate from solution of strain-based model.

        :param sol: Solution of strain-based model.
        :type sol: NDArray (n x ndf)
        :return: Effective shear rate.
        :rtype: NDArray (n x 1)
        :raises NotImplementedError: If effective shear rate computation is not implemented in subclasses.
        """

        raise NotImplementedError('Effective shear rate computation has to be implemented in subclasses.')

    def _compute_D_from_eig(self, lam: NDArray) -> NDArray:
        """
        Compute cell distortion D = (L-B)/(L+B) from eigenvalues of morphology tensor.

        :param lam: Nodal morphology eigenvalues lambda.
        :type lam: NDArray (n x 3)
        :return: Nodal cell distortion values.
        :rtype: NDArray (n x 1)
        """

        lambSqrt = np.sqrt(lam)
        L = np.amax(lambSqrt,1)
        B = np.amin(lambSqrt,1)
        D = np.divide(L-B,L+B)
        return D
    
    def _compute_Geff_from_D(self, D: NDArray) -> NDArray:
        """
        Compute effective strain rate G_eff from cell distortion D.

        :param D: Nodal cell distortion values.
        :type D: NDArray
        :return: Solution field for effective shear rate.
        :rtype: NDArray
        """

        f1 = self._coeffs.f1
        f2 = self._coeffs.f2
        Geff = np.divide(2*f1*D, f2*(np.ones_like(D)-np.multiply(D,D)))
        return Geff
    
    def _compute_strain_vort_rot(self, t: float) -> Tuple[Tensor3, Tensor3, Tensor3]:
        """
        Compute strain, vorticity, and rotation at time t.

        :param t: Time.
        :type t: float
        :return: Tuple of strain tensor, vorticity tensor, and angular velocity of frame of reference.
        :rtype: Tuple[Tensor3, Tensor3, Tensor3]
        """

        dv_i = self._dv(t).reshape(3, 3).T
        E = 0.5 * (dv_i + dv_i.T)
        W = 0.5 * (dv_i - dv_i.T)
        Om = self._unpack_antisymmetric(self._omega(t))

        return (E, W, Om)
    
    def _unpack_antisymmetric(self, vec: Vector3) -> Tensor3:
        """
        Unpack values of antisymmetric vector into tensor.

        :param vec: Vector to unpack.
        :type vec: Vector3
        :return: Antisymmetric tensor.
        :rtype: Tensor3
        """

        return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])
    
    def _RHS(self, t: float, y: NDArray) -> NDArray:
        """
        Right-hand side of ODE for strain-based model. Has to be implemented for each strain-based model.

        :param t: Time.
        :type t: float
        :param y: Current state.
        :type y: NDArray
        :return: Right-hand side of ODE.
        :rtype: NDArray
        :raises NotImplementedError: If RHS is not implemented in subclasses.
        """

        raise NotImplementedError('RHS has to be implemented in subclasses.')
    
    def _compute_steady_state_shear(self, t: float) -> Tuple[Vector3, Tensor3]:
        """
        Computes an approximate steady state. This is done by reducing the three-dimensional strain tensor to a scalar shear rate first using the second tensor invariant. Then the steady state for a simple shear flow of that intensity is computed. The equations for steady state in shear flow with f2 != f3 have been derived by myself on a piece of paper. They are not published anywhere. The equations for f2 = f3 are available in Arora et al. (2004).

        :param t: Time.
        :type t: float
        :return: Tuple of eigenvalues of morphology tensor and basis vectors of morphology tensor.
        :rtype: Tuple[Vector3, Tensor3]
        """

        # Compute shear rate.
        E = self._compute_strain_tensor(t)
        G = self._compute_shear_rate_second_invariant(E)

        f1 = self._coeffs.f1
        f2 = self._coeffs.f2
        f3 = self._coeffs.f3

        # Initialize the basis vectors.
        EV_2 = np.array([0, 0, 1])

        # Calculate eigenvalues in steady state.
        phi = (1 - f2 ** 2 * G ** 2 / (f1 ** 2 + f3 ** 2 * G ** 2)) ** (1/3) # convenience variable
        lambda_2 = phi
        lambda_1 = phi * (f1 ** 2 + G ** 2 * f3 ** 2 - G * f2 * np.sqrt(G ** 2 * f3 ** 2 + f1 ** 2)) / (f1 ** 2 - G ** 2 * (f2 ** 2 - f3 ** 2))
        lambda_3 = phi * (f1 ** 2 + G ** 2 * f3 ** 2 + G * f2 * np.sqrt(G ** 2 * f3 ** 2 + f1 ** 2)) / (f1 ** 2 - G ** 2 * (f2 ** 2 - f3 ** 2))
        lamb = np.array([lambda_1, lambda_2, lambda_3])
        
        # Calculate the other basis vectors.
        EV_1 = np.array([(G * f3 - np.sqrt(f1 ** 2 + (G * f3) ** 2)) / f1, 1, 0])
        EV_3 = np.array([(G * f3 + np.sqrt(f1 ** 2 + (G * f3) ** 2)) / f1, 1, 0])
        
        # Normalize the basis vectors.
        EV_1 = EV_1 / np.linalg.norm(EV_1)
        EV_3 = EV_3 / np.linalg.norm(EV_3)

        # Construct the matrix from the basis vectors.
        Q = np.column_stack((EV_1, EV_2, EV_3))
        if np.linalg.det(Q) < 0: # make sure that Q is a rotation matrix, not a reflection
            Q[:, 2] = -Q[:, 2]
        
        return lamb, Q

class MorphologyTensorFormulation(StrainBasedModel):
    """Abstract base class that represents any morphology model derived from the Arora model that uses the morphology tensor :code:`S = [S_11, S_22, S_33, S_12, S_13, S_23]` as solution variable."""

    def _compute_eig_from_morph(self, S: NDArray) -> NDArray:
        """
        Compute eigenvalues of morphology tensor.

        :param S: Morphology tensor solution, assumed to be written as vector in form [S_11, S_22, S_33, S_12, S_13, S_23].
        :type S: NDArray
        :return: Eigenvalues at nodes.
        :rtype: NDArray
        """

        if np.shape(S)[1] == 3:     # 2D -- simple formula
            d = S[:,0]
            e = S[:,1]
            f = S[:,2]

            de = d+e
            tmp1 = de*0.5
            tmp2 = 0.5*np.sqrt( np.multiply(de,de) - 4*(np.multiply(d,e) - np.multiply(f,f)) )
            l1 =  tmp1 + tmp2
            l2 =  tmp1 - tmp2
        
            lam = np.stack((l1,l2), axis=1)
        else:
            nn = np.shape(S)[0]
            ndf = np.shape(S)[1]
            assert ndf == 6
            lam = np.zeros((nn,3))
            for (i, S_i) in enumerate(S):
                S_mat = self._unpack_morphology(S_i)
                lam[i,:] = np.flip(np.linalg.eigvalsh(S_mat))

        return lam

    def _pack_morphology(self, S: NDArray) -> NDArray:
        """
        Pack values from symmetric morphology tensor into vector.

        :param S: Symmetric morphology tensor.
        :type S: NDArray
        :return: Vector of morphology tensor values.
        :rtype: NDArray
        """

        morph_vec = np.zeros((6,))
        morph_vec[0] = S[0, 0]
        morph_vec[1] = S[1, 1]
        morph_vec[2] = S[2, 2]
        morph_vec[3] = S[0, 1]
        morph_vec[4] = S[0, 2]
        morph_vec[5] = S[1, 2]
        return morph_vec
    
    def _unpack_morphology(self, morph_vec: NDArray) -> Tensor3:
        """
        Unpack values from vector into symmetric morphology tensor.

        :param morph_vec: Vector of morphology tensor values.
        :type morph_vec: NDArray
        :return: Symmetric morphology tensor.
        :rtype: Tensor3
        """

        S = np.zeros((3, 3))
        S[0, 0] = morph_vec[0]
        S[1, 1] = morph_vec[1]
        S[2, 2] = morph_vec[2]
        S[0, 1] = morph_vec[3]
        S[1, 0] = morph_vec[3]
        S[0, 2] = morph_vec[4]
        S[2, 0] = morph_vec[4]
        S[1, 2] = morph_vec[5]
        S[2, 1] = morph_vec[5]
        return S

    def _initial_condition_undeformed(self) -> NDArray:
        """
        Set initial condition for undeformed cells in morphology tensor models, i.e., S = I.

        :return: Initial condition for undeformed cells.
        :rtype: NDArray
        """

        return np.array([1, 1, 1, 0, 0, 0])
    
    def _initial_condition_steady(self) -> NDArray:
        """
        Set initial condition for steady shear in morphology tensor models.

        :return: Initial condition for steady shear.
        :rtype: NDArray
        """

        lamb, Q = self._compute_steady_state_shear(self._t0)
        S = np.matmul(Q, np.matmul(np.diag(lamb), Q.T)) # S = Q * lambda * Q^T (eigendecomposition)
        return self._pack_morphology(S)
    
    def _compute_Geff_from_sol(self, sol: NDArray) -> NDArray:
        """
        Compute effective shear rate from solution of morphology tensor model.

        :param sol: Solution of morphology tensor model, assumed to be written as vector in form [S_11, S_22, S_33, S_12, S_13, S_23].
        :type sol: NDArray (n x 6)
        :return: Effective shear rate.
        :rtype: NDArray (n x 1)
        """

        lam = self._compute_eig_from_morph(sol)
        D = self._compute_D_from_eig(lam)
        Geff = self._compute_Geff_from_D(D)
    
        return Geff
    
    def _recovery_term(self, S: Tensor3) -> Tensor3:
        """
        Compute morphology recovery term for models that include morphology tensor.

        :param S: Morphology tensor.
        :type S: Tensor3
        :return: Morphology recovery term.
        :rtype: Tensor3
        """
        
        g = 3*np.linalg.det(S) / self._compute_second_invariant(S)
        return -( S - g * np.eye(3) )
    
    def _strain_term(self, E: Tensor3, S: Tensor3) -> Tensor3:
        """
        Compute strain source term for models that include morphology tensor.

        :param E: Strain tensor.
        :type E: Tensor3
        :param S: Morphology tensor.
        :type S: Tensor3
        :return: Strain source term.
        :rtype: Tensor3
        """

        return np.matmul(E, S) + np.matmul(S, E)
    
    def _vorticity_term(self, W: Tensor3, S: Tensor3) -> Tensor3:
        """
        Compute vorticity source term for models that include morphology tensor.

        :param W: Vorticity tensor.
        :type W: Tensor3
        :param S: Morphology tensor.
        :type S: Tensor3
        :return: Vorticity source term.
        :rtype: Tensor3
        """

        return np.matmul(W, S) - np.matmul(S, W)

class AroraSimplified(MorphologyTensorFormulation):
    r"""
    Represents the simplified Eulerian morphology model (see :ref:`arora-simplified-model`)
    """

    def get_name(self) -> str:
        """
        Get name of blood damage model.

        :return: Name of the blood damage model.
        :rtype: str
        """
        return 'simplified-arora'
    
    def _RHS(self, t: float, y: NDArray) -> NDArray:
        """
        Compute the right-hand side of the ODE for the cell model.

        :param t: Time.
        :type t: float
        :param y: Current state.
        :type y: NDArray
        :return: Right-hand side of the ODE.
        :rtype: NDArray
        """

        E, W, Om = self._compute_strain_vort_rot(t)

        S = self._unpack_morphology(y)

        f1 = self._coeffs.f1
        f2 = self._coeffs.f2
        f3 = self._coeffs.f3

        rec = self._recovery_term(S)
        str = self._strain_term(E,S)
        vort = self._vorticity_term(W,S)
        rot = self._vorticity_term(Om,S)

        f_S = f1*rec + f2*str + f3*vort - rot
        f = self._pack_morphology(f_S)

        return f
    
class MorphologyEigFormulation(StrainBasedModel):
    r"""Abstract superclass for all morphology models derived from the Arora model that employ a formulation explicit in the eigenvalues :math:`(\lambda_1, \lambda_2, \lambda_3)`"""

    def _compute_Geff_from_sol(self, sol: NDArray) -> NDArray:
        r"""
        Compute effective shear rate from solution of strain-based model.

        :param sol: Solution of strain-based model, assumed to be written as vector in form :math:`(\lambda_1, \lambda_2, \lambda_3, \dots)`.
        :type sol: NDArray (n x ndf)
        :return: Effective shear rate.
        :rtype: NDArray (n x 1)
        """

        lam = sol[:,:3] # eigenvalues are first three entries
        D = self._compute_D_from_eig(lam)
        Geff = self._compute_Geff_from_D(D)
        return Geff

    def _dLdt(self, lamb: Vector3, Et: Tensor3) -> Vector3:
        """
        Computes source term for eigenvalues.

        :param lamb: Eigenvalues.
        :type lamb: Vector3
        :param Et: Transformed strain tensor.
        :type Et: Tensor3
        :return: Source term for eigenvalues.
        :rtype: Vector3
        """

        f1 = self._coeffs.f1
        f2 = self._coeffs.f2
        III = np.prod(lamb)
        II = lamb[0] * lamb[1] + lamb[1] * lamb[2] + lamb[0] * lamb[2]
        f = -f1 * (lamb - 3 * III / II) + 2 * f2 * np.diag(Et) * lamb
        return f

class AroraFullEig(MorphologyEigFormulation):
    r"""
    Represents the eigenvalue-eigenvector formulation of the full Eulerian formulation of the Arora model (see :ref:`arora-full-eig-model`)
    """

    def get_name(self) -> str:
        """
        Get name of blood damage model.

        :return: Name of the blood damage model.
        :rtype: str
        """

        return 'full-arora'
    
    def _initial_condition_undeformed(self) -> Vector12:
        """
        Set initial condition for undeformed cells. There is some singularity in the model if the initial condition is exactly undeformed. Therefore, we add a small perturbation.

        :return: Initial condition for undeformed cells with a small perturbation to avoid singularity in the model.
        :rtype: Vector12
        """

        # Add perturbation to avoid singularity.
        eps = 1e-3
        lamb = np.array([1.0-eps, 1.0, 1.0+eps])
        Q = np.eye(3) 
        Q_lin = Q.reshape((9,))
        return np.concatenate((lamb, Q_lin))
    
    def _initial_condition_steady(self) -> Vector12:
        """
        Set initial condition to steady shear state.

        :return: Steady shear state, expressed in model variables.
        :rtype: Vector12
        """

        lamb, Q = self._compute_steady_state_shear(self._t0)
        Q_lin = Q.reshape((9,))
        return np.concatenate((lamb, Q_lin))
    
    def _RHS(self, t: float, y: Vector12) -> Vector12:
        """
        Compute the right-hand side of the ODE.

        :param t: Time.
        :type t: float
        :param y: Current state.
        :type y: Vector12
        :return: Right-hand side of the ODE.
        :rtype: Vector12
        """

        E, W, Om = self._compute_strain_vort_rot(t)
        W = W - Om

        # Get eigenvalues and eigenvectors from solution vector.
        lamb = y[:3]
        Q_lin = y[3:]

        Q = self._orthonormalize(Q_lin.reshape((3, 3))) # orthonormalize eigenvectors

        # Compute transformed strain and vorticity tensors.
        Et = np.matmul(Q.T, np.matmul(E, Q))
        Wt = np.matmul(Q.T, np.matmul(W, Q))

        # Compute source terms.
        f_lamb = self._dLdt(lamb, Et)
        f_Q = self._dQdt(lamb, Et, Wt, Q)

        # Pack source terms into vector.
        f_Q_lin = f_Q.reshape((9,))
        f = np.concatenate((f_lamb, f_Q_lin))

        return f
    
    def _dQdt(self, lamb: Vector3, Et: Tensor3, Wt: Tensor3, Q: Tensor3) -> Tensor3:
        """
        Computes source term for orientation tensor Q.

        :param lamb: Eigenvalues.
        :type lamb: Vector3
        :param Et: Transformed strain tensor.
        :type Et: Tensor3
        :param Wt: Transformed vorticity tensor.
        :type Wt: Tensor3
        :param Q: Eigenvectors.
        :type Q: Tensor3
        :return: Source term for orientation tensor.
        :rtype: Tensor3
        """
        
        f2t = self._coeffs.f2t
        f3t = self._coeffs.f3t
        Omt = np.zeros((3, 3))
        for j in range(3):
            for k in range(j + 1, 3):
                Omt[j, k] = f2t * Et[j, k] * (lamb[k] + lamb[j]) / (lamb[k] - lamb[j]) + f3t * Wt[j, k]
                Omt[k, j] = -Omt[j, k]
        f = np.matmul(Q, Omt)
        return f

    def _orthonormalize(self, EV: Tensor3) -> Tensor3:
        """
        Orthonormalizes eigenvectors.

        :param EV: Eigenvectors.
        :type EV: Tensor3
        :return: Orthonormalized eigenvectors.
        :rtype: Tensor3
        """

        Q = np.zeros_like(EV)
        for i in range(EV.shape[1]):
            Q[:, i] = EV[:, i]

            # orthogonalize
            for j in range(i):
                Q[:, i] = Q[:, i] - np.matmul(Q[:, i].T, Q[:, j]) * Q[:, j]

            # normalize
            Q[:, i] = Q[:, i] / np.sqrt(np.matmul(Q[:, i].T, Q[:, i]))

        return Q
    
class TankTreading(MorphologyEigFormulation):
    r"""
    Represents the tank-treading cell deformation model (see :ref:`tanktreading-model`), 
    """

    def get_name(self) -> str:
        """
        Get name of blood damage model.

        :return: Name of the blood damage model.
        :rtype: str
        """

        return 'tank-treading'
    
    def _initial_condition_undeformed(self) -> Vector3:
        """
        Set initial condition for undeformed cells.

        :return: Initial condition for undeformed cells.
        :rtype: Vector3
        """

        lamb = np.array([1, 1, 1])
        return lamb
    
    def _initial_condition_steady(self) -> Vector3:
        """
        Set initial condition to steady shear state.

        :return: Steady shear state, expressed in model variables.
        :rtype: Vector3
        """

        lamb, Q = self._compute_steady_state_shear(self._t0)
        return lamb
    
    def _RHS(self, t: float, y: Vector3) -> Vector3:
        """
        Compute the right-hand side of the ODE for the tank-treading model.

        :param t: Time.
        :type t: float
        :param y: Current state.
        :type y: Vector3
        :return: Right-hand side of the ODE.
        :rtype: Vector3
        """

        E, W, Om = self._compute_strain_vort_rot(t)
        W = W - Om

        # Get eigenvalues from solution vector.
        lamb = y[:3]

        # Get orientation from algebraic equation.
        Q = self._getSteadyOrientation3D(lamb, E, W)

        # Compute transformed strain tensor.
        Et = np.matmul(Q.T, np.matmul(E, Q))

        f = self._dLdt(lamb, Et)
        return f
    
    def _getSteadyOrientation2D(self, Eii: float, Eij: float, Ejj: float, 
                                Wij: float, li: float, lj: float) -> float | None:
        """
        Computes the steady orientation for a 2D ellipse.

        :param Eii: Diagonal element of (symmetric) strain tensor.
        :type Eii: float
        :param Eij: Off-diagonal element of (symmetric) strain tensor.
        :type Eij: float
        :param Ejj: Other diagonal element of (symmetric) strain tensor.
        :type Ejj: float
        :param Wij: Off-diagonal element of (antisymmetric) vorticity tensor.
        :type Wij: float
        :param li: Eigenvalue of morphology tensor.
        :type li: float
        :param lj: Other eigenvalue of morphology tensor.
        :type lj: float
        :return: Angle for steady orientation. None, if no steady orientation exists.
        :rtype: float | None
        """
        
        tol = 1e-7

        if np.abs(lj - li) < tol:
            thetaStar = -0.5 * np.arctan(2 * Eij / (Eii - Ejj))
            return thetaStar

        dist = (lj + li) / (lj - li)

        A = 0.5 * (Eii - Ejj) * dist
        B = Eij * dist
        C = Wij
        BmC = B - C

        discr = A * A + B * B - C * C

        if discr >= 0:  # steady state exists
            if np.abs(BmC) < tol:
                if A >= -tol:
                    thetaStar = np.pi * 0.5
                else:
                    thetaStar = -np.arctan(B / A)
            else:
                thetaStar = np.arctan((A + np.sqrt(discr)) / BmC)
        else:   # steady state does not exist
            thetaStar = None

        return thetaStar

    def _getSteadyOrientation3D(self, lamb: Vector3, E: Tensor3, W: Tensor3, tol: float=1e-5) -> Tensor3:
        """
        Computes the steady orientation for a 3D ellipsoid. The algorithm is presented in Dirkes et al. (2023). 
        Warning: in contrast to the paper, we employ the opposite order of eigenvalues, i.e., lambda_1 <= lambda_2 <= lambda_3. 
        This can lead to some index differences compared to the algorithm in the paper.

        :param lamb: Eigenvalues of morphology tensor.
        :type lamb: Vector3
        :param E: Strain tensor.
        :type E: Tensor3
        :param W: Vorticity tensor.
        :type W: Tensor3
        :param tol: Tolerance for convergence. Defaults to 1e-5.
        :type tol: float
        :return: Orientation tensor.
        :rtype: Tensor3
        """

        it = 0
        maxIt = 100

        # Sort eigenvalues in ascending order and switch eigenvectors accordingly.
        idx_l = np.argsort(lamb)
        lamb = lamb[idx_l]
        [D,V] = np.linalg.eig(E)
        idx_s = np.argsort(D)
        V_sorted = V[:, idx_s[idx_l]]
        if np.linalg.det(V_sorted) < 0:
            V_sorted[:, 2] = -V_sorted[:, 2]
        Q = V_sorted
        Et = np.matmul(Q.T, np.matmul(E, Q))
        Wt = np.matmul(Q.T, np.matmul(W, Q))

        converged = False

        while (not converged) and (it < maxIt):

            it += 1

            # Automatically adjust relaxation factor if convergence is slow.
            if(it < 10):
                relaxFact = 1.0
            elif(it < 100):
                relaxFact = 0.1
            else:
                relaxFact = 0.01
            
            converged = True

            # Loop over all axes.
            for iax in range(3):

                # Get indices for projection of tensor on axis iax.
                i, j = self._getIdxForAxis(iax)
                Eij = Et[i, j]
                Eii = Et[i, i]
                Ejj = Et[j, j]
                Wij = Wt[i, j]
                li = lamb[i]
                lj = lamb[j]

                # Compute steady orientation in 2D.
                thetaStar = self._getSteadyOrientation2D(Eii, Eij, Ejj, Wij, li, lj)

                # If no steady orientation exists, stop here.
                if thetaStar is None:
                    return np.zeros_like(Q)

                # If steady orientation exists, rotate morphology tensor.
                R = self._getRotMat(iax, relaxFact * thetaStar)
                Et = np.matmul(R.T, np.matmul(Et, R))
                Wt = np.matmul(R.T, np.matmul(Wt, R))

                # Update orientation.
                Q = np.matmul(Q, R)

                # Check convergence.
                converged = converged and np.abs(thetaStar) <= tol

        return Q

    def _getIdxForAxis(self, iax: int) -> Tuple[int, int]:
        """
        Get indices for projection of tensor on axis iax.

        :param iax: Axis.
        :type iax: int
        :return: Two indices (i,j) to project tensor on axis iax.
        :rtype: Tuple[int, int]
        """

        if iax == 0:
            return 2, 1
        elif iax == 1:
            return 0, 2
        else:
            return 1, 0

    def _getRotMat(self, iax: int, theta: float) -> Tensor3:
        """
        Get rotation matrix for rotation around axis iax by angle theta.

        :param iax: Axis.
        :type iax: int
        :param theta: Angle.
        :type theta: float
        :return: Rotation matrix.
        :rtype: Tensor3
        """

        c = np.cos(theta)
        s = np.sin(theta)
        if iax == 0:
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif iax == 1:
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else:
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

class TankTreadingRotationCorrection(TankTreading):
    """
    Represents the tank-treading model (Dirkes et al. 2023) with a correction term for cell rotation along the pathline. This model is still experimental and not yet published.
    """

    _x: Callable[[float], Vector3]  
    """
    Relevant position measure as a function of pathline time, e.g., absolute coordinates 
    or distance from rotational center (required only for tank-treading with pathline correction).
    """

    _v: Callable[[float], Vector3]
    """
    Velocity as a function of pathline time 
    (required only for tank-treading with pathline correction).
    """

    def get_name(self) -> str:
        """
        Get name of blood damage model.

        :return: Name of the blood damage model.
        :rtype: str
        """

        return 'tank-treading-pathline'
    
    def set_time_dependent_quantitites(self, t0: float, tend: float, dv: Callable[[float], Vector9] | None = None, omega: Callable[[float], Vector3] | None = None, x: Callable[[float], Vector3] | None = None, v: Callable[[float], Vector3] | None = None) -> None:
        """
        Set time-dependent quantities for the cell model. Overrides superclass method to include position measure x and velocity v.

        :param t0: Start time.
        :type t0: float
        :param tend: End time.
        :type tend: float
        :param dv: Time-dependent velocity gradient dv/dt.
        :type dv: Callable[[float], Vector9]
        :param omega: Time-dependent angular velocity.
        :type omega: Callable[[float], Vector3]
        :param x: Time-dependent position measure, e.g., absolute coordinates or distance from rotational center. 
        :type x: Callable[[float], Vector3]
        :param v: Time-dependent velocity.
        :type v: Callable[[float], Vector3
        :raises ValueError: If position measure x or velocity v is not provided.
        """
        
        # Set same quantities as superclass.
        super().set_time_dependent_quantitites(t0, tend, dv, omega, x, v)

        # Additionally set position measure and velocity.
        if x is None:
            raise ValueError('Position measure x is required for tank-treading with pathline correction.')
        
        if v is None:
            raise ValueError('Velocity v is required for tank-treading with pathline correction.')
        
        self._x = x
        self._v = v

    def _compute_strain_vort_rot(self, t: float) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Overrides _compute_strain_vort_rot from superclass to include correction term for cell rotation along the pathline.

        :param t: Time.
        :type t: float
        :return: Strain tensor, Vorticity tensor, Angular velocity of frame of reference, including correction term.
        :rtype: Tuple[NDArray, NDArray, NDArray]
        """

        # Compute correction term.
        r = self._x(t) # orthogonal distance from center
        v = self._v(t) # advection velocity

        r_norm = np.linalg.norm(r)**2
        if r_norm < 1e-8: # avoid division by zero
            om_correction = np.zeros(3)
        else:
            om_correction = np.cross(r, v) / r_norm

        # Compute strain and vorticity as before.
        E, W, Om = super()._compute_strain_vort_rot(t)

        # Add correction term.
        Om = Om + self._unpack_antisymmetric(om_correction)

        return (E, W, Om)