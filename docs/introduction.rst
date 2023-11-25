Introduction
============

This package is meant to provide a simple way to post-process CFD simulation data for medical devices with the objective of predicting hemolysis, i.e., red blood cell damage. In particular, it supports the following functionalities:

1. Reading in CFD results
2. Computing velocity gradients
3. Determining trajectories (pathlines) of red blood cells (RBCs)
4. Interpolating arbitrary other fields to the RBC pathlines
5. Computing various stress-based and strain-based measures of RBC shear
6. Integrating various empirical hemolysis correlations along the RBC pathlines using the aforementioned measures

The package is written in Python and uses `VTK <https://vtk.org/>`_ to process the CFD data. As a result, the package is compatible with any CFD solver that can export data in the VTK format. 
I am currently using it with our in-house finite-element solver :code:`XNS`. I am further planning on extending compatibility to :code:`.vtu` files (COMSOL) and :code:`.cgns` files (ANSYS Fluent).

The package is further meant for steady flow data. Rotational flows such as those present in most blood pumps are meant to be handled using the MRF approach. Transient simulations need to be averaged in time to obtain quasi-MRF data before being processed by this package.