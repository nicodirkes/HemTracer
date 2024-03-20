Theoretical Background
=======================

This section aims to describe the physical models implemented in this package. 

.. _rbc-models:

RBC models
----------

RBC models aim to describe the effects of the three-dimensional fluid forces on the red blood cells (RBCs). All of them compute a representative scalar shear rate :math:`G_s` along the pathline that represents the cell distortion state. This shear rate may then be used to compute the index of hemolysis (see :ref:`hemolysis-models`). This can be done using two different approaches: :ref:`strain-based-models` and :ref:`stress-based-models`.

.. _strain-based-models:

Strain-based models
~~~~~~~~~~~~~~~~~~~

Strain-based models explicitly resolve cell deformation in response to the fluid forces. They are based on an ODE system that describes the evolution of the cell shape. The shear rate is then computed from the cell deformation (see Arora et al. :cite:p:`d_arora_tensor-based_2004`). 

.. _arora-full-eig-model:

Full Arora model
^^^^^^^^^^^^^^^^

This model was introduced by Arora et al. :cite:p:`d_arora_tensor-based_2004`. Due to some numerical issues with the original formulation, we perform a spectral decomposition, as described in our paper :cite:p:`dirkesEulerianFormulationTensorBased`. The model is thus explicit in the eigenvalues :math:`(\lambda_1, \lambda_2, \lambda_3)` and eigenvectors :math:`\mathbf{Q} = [ \mathbf{v}_1 | \mathbf{v}_2 | \mathbf{v}_3]`. 
The governing equations are:

.. math:: 
    :nowrap:

    \begin{eqnarray}
    \frac{\mathrm d \lambda_i }{ \mathrm dt}  & = & -f_1 \left( \lambda_i - \frac{3 \, \lambda_1 \lambda_2 \lambda_3}{\lambda_1 \lambda_2 + \lambda_2 \lambda_3 + \lambda_1 \lambda_3} \right) + 2 f_2 \tilde{E}_{ii} \lambda_i \, ,
    \\
    \frac{\mathrm d \mathbf{Q} }{ \mathrm dt}  & = & \mathbf{Q} \tilde{\mathbf{\Omega}} \, , \qquad \mathrm{with} \quad \tilde{\Omega}_{ij} = \tilde{f_2} \tilde{E}_{ij} \frac{\lambda_j + \lambda_i}{\lambda_j - \lambda_i} + \tilde{f_3} \tilde{W}_{ij} \, .
    \end{eqnarray}

Here, :math:`\tilde{\mathbf{E}} = \mathbf{Q}^\mathrm{T} \mathbf{E} \mathbf{Q}` and :math:`\tilde{\mathbf{W}} = \mathbf{Q}^\mathrm{T} \mathbf{W} \mathbf{Q}` are the transformed strain and vorticity tensors, respectively.

This is implemented as :class:`hemtracer.rbc_model.strain_based.AroraFullEig`. The default coefficients, i.e., those corresponding to the original Arora model, are :math:`f_1 = 5.0 \, \mathrm{s}^{-1}`, :math:`f_2 = 4.2298 \cdot 10^{-4}`, :math:`\tilde{f_2} = \tilde{f_3} = 1.0`.

.. _arora-simplified-model:

Simplified Arora model
^^^^^^^^^^^^^^^^^^^^^^

This model was derived by Pauli et al. :cite:p:`pauli_transient_2013` by neglecting the rotation of the eigenvectors in the Arora model :cite:p:`d_arora_tensor-based_2004`. The model is formulated in terms of the symmetric morphology tensor :math:`\mathbf{S} \in \mathbb{R}^{n \times n}`. The governing equation is

.. math:: \frac{\mathrm d \mathbf{S}}{\mathrm dt} = -f_1 (\mathbf{S} - g(\mathbf{S}) \mathbf{I}) + f_2 (\mathbf{ES} + \mathbf{SE}) + f_3 (\mathbf{WS} - \mathbf{SW}) \, ,

with the strain tensor :math:`\mathbf{E} = \frac{1}{2}((\boldsymbol{\nabla} \mathbf{v}) + (\boldsymbol{\nabla} \mathbf{v})^\mathrm{T})` and vorticity tensor :math:`\mathbf{W} = \frac{1}{2} (\boldsymbol{\nabla} \mathbf{v} - (\boldsymbol{\nabla} \mathbf{v})^{\mathrm{T}})`.

This is implemented as :class:`hemtracer.rbc_model.strain_based.AroraSimplified`.

.. _tanktreading-model:

Tank-treading model
^^^^^^^^^^^^^^^^^^^

This model was derived by Dirkes et al. :cite:p:`dirkesEulerianFormulationTensorBased` by replacing the differential equation for the orientation tensor Q by an algebraic equation for equilibrium orientation. The model is generally more efficient than the full Eulerian reformulation and more robust, as it contains only the eigenvalues (3 DoF's). The governing equations are

.. math::
    :nowrap:

    \begin{eqnarray}
    \frac{\mathrm d \lambda_i }{ \mathrm dt}  & = & -f_1 \left( \lambda_i - \frac{3 \, \lambda_1 \lambda_2 \lambda_3}{\lambda_1 \lambda_2 + \lambda_2 \lambda_3 + \lambda_1 \lambda_3} \right) + 2 f_2 \tilde{E}_{ii} \lambda_i \, ,
    \\
    \mathbf Q & = &
    \begin{cases}
        \mathbf Q_\star & \text{tank-treading} \\
        \mathbf 0 & \text{tumbling}
    \end{cases} \, , \qquad \mathrm{with} \quad
    \tilde{\mathbf{E}} = \mathbf{Q}^\mathrm{T} \mathbf{E} \mathbf{Q} \, ,
    \end{eqnarray}

This is implemented as :class:`hemtracer.rbc_model.strain_based.TankTreading`.

.. _stress-based-models:

Stress-based models
~~~~~~~~~~~~~~~~~~~

Stress-based models do not explicitly resolve cell deformation. Instead, they compute the shear rate from the instantaneous fluid strain rate tensor :math:`\mathbf{E}`. 

.. _bludszuweit-model:

Bludszuweit model
^^^^^^^^^^^^^^^^^^

This model was proposed by Bludszuweit :cite:p:`bludszuweitModelGeneralMechanical1995a`. It computes a representative scalar from instantaneous fluid strain, similar to the von Mises stress:

.. math:: G_s = \frac{2}{\sqrt{3}} \sqrt{
    \left(E_{xx}^2 + E_{yy}^2 + E_{zz}^2\right)
    - \left(E_{xx} E_{yy} + E_{xx} E_{zz} + E_{yy} E_{zz}\right)
    + 3 \left(E_{xy}^2 + E_{xz}^2 + E_{yz}^2\right)}

This is implemented as :class:`hemtracer.rbc_model.stress_based.Bludszuweit`.

.. _faghih-sharp-model:

Faghih and Sharp model
^^^^^^^^^^^^^^^^^^^^^^

This model was proposed by Faghih and Sharp :cite:p:`faghih_deformation_2020`. It weighs extensional and shear stresses differently:

.. math:: G_s = 2 \sqrt{C_n^2 \left[ E_{xx}^2 + E_{yy}^2 + E_{zz}^2 
                                    - (E_{xx} E_{yy} + E_{xx} E_{zz} + E_{yy} E_{zz}) \right]
                    + E_{xy}^2 + E_{xz}^2 + E_{yz}^2}

With :math:`\sqrt{3} C_n = 33.79`. This is implemented as :class:`hemtracer.rbc_model.stress_based.FaghihSharp`. 

.. _frobenius-model:

Frobenius norm
^^^^^^^^^^^^^^^

Computes a representative scalar from instantaneous fluid strain using the Frobenius norm:

.. math:: G_s = \sqrt{2 \sum_{i,j} E_{ij}^2}

This is implemented as :class:`hemtracer.rbc_model.stress_based.Frobenius`.

.. _second-invariant-model:

Second strain invariant
^^^^^^^^^^^^^^^^^^^^^^^

Computes a representative scalar from instantaneous fluid strain using the second strain invariant:

.. math:: G_s = \sqrt{ 2 \mathrm{tr}(\mathbf{E}^2 ) }

This is implemented as :class:`hemtracer.rbc_model.stress_based.SecondInvariant`.


.. _hemolysis-models:

Hemolysis models
----------------

Hemolysis models employ an empirical correlation between the scalar shear rate :math:`G_s` and the index of hemolysis :math:`IH \, [\%]`. Commonly, this correlation takes the form of a power law:

.. math:: IH = A_\mathrm{Hb} (\mu G_s)^{\alpha_\mathrm{Hb}} t^{\beta_\mathrm{Hb}} \, .

This is implemented as :class:`hemtracer.hemolysis_model.PowerLaw`. 

The parameter :math:`\mu` represents the viscosity of blood. It is usually assumed to be constant and equal to :math:`3.5 \, \mathrm{mPa \cdot \mathrm{s}`. The parameter :math:`t` represents exposure time. It is integrated along the pathline, along with the shear rate :math:`G_s`.

Coefficients
~~~~~~~~~~~~~~~~~~~

The coefficients :math:`A_\mathrm{Hb}`, :math:`\alpha_\mathrm{Hb}` and :math:`\beta_\mathrm{Hb}` are determined empirically. Over the past 30 years, several studies have found a wide range of possible values. The available correlations are given in :class:`hemtracer.hemolysis_model.IHCorrelation`. 


Numerical integration
~~~~~~~~~~~~~~~~~~~~~

The index of hemolysis is computed by numerically integrating the power law along the pathline. There is some discussion in literature on the discretization of the power law. Various approaches are presented by Taskin et al. :cite:p:`taskinEvaluationEulerianLagrangian2012`. Note that they use a different definition of the power law:

.. math:: HI = C t^\alpha \sigma^\beta \, .

They compiled five approaches to integrate this power law. They can be selected in the constructor of :class:`hemtracer.hemolysis_model.PowerLaw` by setting the option `integration_scheme`. The available options are:

+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option               | Equation                                                                                                                                                                          |
+======================+===================================================================================================================================================================================+
| :code:`basic`        | :math:`HI1(t_n) = \sum_{i=1}^n C (\Delta t)_i^\alpha \sigma_i^\beta`                                                                                                              |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :code:`timeDiff`     | :math:`HI2(t_n) = \sum_{i=1}^n C \alpha t_i^{\alpha-1} \sigma_i^\beta (\Delta t)_i`                                                                                               |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :code:`linearized`   | :math:`HI3(t_n) = C \left( \sum_{i=1}^n (\Delta t)_i \sigma_i^{\beta/\alpha} \right)`                                                                                             |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :code:`mechDose`     | :math:`HI4(t_n) = \sum_{i=1}^n \alpha C \left( \sum_{j=1}^i \sigma_j^{\beta/\alpha} \right)^{\alpha-1} \sigma_i^{\beta/\alpha} \Delta t_i`                                        |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :code:`effTime`      | :math:`HI5(t_n) = C (t_\mathrm{eff}^n + (\Delta t)_n)^{\alpha} \sigma^\beta \, , \qquad  t_\mathrm{eff}^n = \left( \frac{HI5(t_{n-1})}{C \sigma^\beta} \right)^{1/\alpha}`        |
+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+