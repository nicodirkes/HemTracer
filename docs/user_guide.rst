User Guide
=====================================

This guide provides a brief overview of the package and its usage. For more detailed information, refer to the :ref:`API documentation <api doc>`.

Installation
-------------------------------------
The recommended way to install is to clone the `GitHub repository <https://github.com/nicodirkes/HemTracer>`_ and install the package using pip:

.. code-block:: bash

    git clone
    cd HemTracer
    python3 -m pip install .

Note that the package has been tested only with Python 3.12. It may work with older versions, but this is not guaranteed. In addition, the package requires the following dependencies:

- `numpy <https://numpy.org/>`_
- `scipy <https://www.scipy.org/>`_
- `vtk <https://vtk.org/>`_

These dependencies are automatically installed when installing the package using pip. The versions used during development and testing are listed in :file:`requirements.txt` in the repository.

Usage
-------------------------------------
The package is meant to be used as part of a Python script. After installation, it can be imported as follows:

.. code-block:: python

    import hemtracer as ht


Importing CFD data
~~~~~~~~~~~~~~~~~~

The first step is to read in the CFD data. This is handled by the :class:`hemtracer.EulerianFlowField` class. The constructor takes the path to the VTK file as an argument:

.. code-block:: python

    flow_field = ht.EulerianFlowField('path/to/vtk/file.vtk')

If the CFD data represents an MRF simulation (as should be the case for most centrifugal blood pumps), the velocities have to be given all in the absolute frame. The package will then perform the necessary transformations using the :func:`hemtracer.EulerianFlowField.transform_mrf_data` method:

.. code-block:: python

    flow_field.transform_mrf_data(rf_name='ref_frame', rf_rot=2, omega=[0, 100, 0], x0=[0,0,0])

Refer to the documentation of the :func:`hemtracer.EulerianFlowField.transform_mrf_data` method for more details on the arguments. Note that this step is required for all MRF simulations. 

Computing pathlines
~~~~~~~~~~~~~~~~~~~

The next step is to compute pathlines. This is handled by the :class:`hemtracer.PathlineTracker` class. The constructor takes the flow field as an argument:

.. code-block:: python

    pathline_tracker = ht.PathlineTracker(flow_field)

Next, the starting points of the pathlines need to be defined as a list of arrays, e.g., as follows:

.. code-block:: python

    import numpy as np

    x0 = np.asarray([0, 5.0, 0])
    n = 9
    r_arr = np.linspace(0.02,0.2,n) 
    phi_arr = np.linspace(0, 2*np.pi, n, endpoint=False)

    x0_list = []
    for r_i in r_arr:
        for phi_i in phi_arr:
            x0_list.append(x0 + np.asarray([r_i*np.sin(phi_i), 0, r_i*np.cos(phi_i)]))

Then the pathlines can be computed using the :func:`hemtracer.PathlineTracker.compute_pathlines` method:

.. code-block:: python

    pathline_tracker.compute_pathlines(x0_list)


Computing RBC shear
~~~~~~~~~~~~~~~~~~~

In order to compute the index of hemolysis, we first need to compute the effects of the flow forces on the RBCs. This is done by computing a scalar representative shear rate :math:`G_s` along each pathline (see :ref:`rbc-models` for details). First, we need to choose a model to compute this shear rate:

.. code-block:: python

    cell_model = ht.rbc_model.strain_based.TankTreading()

Depending on the type of model, different configuration options are available, e.g., the sampling rate of the flow gradients for stress-based models and ODE solver settings for strain-based models. For details, refer to the :ref:`API documentation of the individual models <rbc models api>`.

In addition, we can choose a model to compute the index of hemolysis from the representative shear rate (see :ref:`hemolysis-models` for details). We will use a power law approach with the correlation from Song et al. :cite:p:`songComputationalFluidDynamics2003`:

.. code-block:: python

    hemolysis_model = ht.hemolysis_model.PowerLawModel( ht.hemolysis_model.IHCorrelation.SONG )

Refer to the documentations of :class:`hemtracer.hemolysis_model.PowerLawModel` and :class:`hemtracer.hemolysis_model.IHCorrelation` for details on the configuration options.

Finally, we construct a :class:`hemtracer.HemolysisSolver` object, which can compute the relevant hemolysis quantities:

.. code-block:: python

    hemolysis_solver = ht.HemolysisSolver(pathline_tracker)
    hemolysis_solver.compute_representativeShear(cell_model)
    hemolysis_solver.compute_hemolysis(cell_model, hemolysis_model)



Sample Script
-------------------------------------

The following script shows a complete example of how to use the package:

.. include:: ../sample/hemo_analysis.py
    :literal: