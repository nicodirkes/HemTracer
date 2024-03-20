Introduction
============

This package is intended to provide a simple way to post-process CFD simulation data for medical devices with the objective of predicting hemolysis, i.e., red blood cell damage. In particular, it supports the following functionalities:

1. Reading in CFD results
2. Computing velocity gradients
3. Determining trajectories (pathlines) of red blood cells (RBCs)
4. Interpolating arbitrary other fields to the RBC pathlines
5. Computing various stress-based and strain-based measures of RBC shear
6. Integrating various empirical hemolysis correlations along the RBC pathlines using the aforementioned measures

The package is written in Python and uses `VTK <https://vtk.org/>`_ to process the CFD data. As a result, the package can import CFD data from any solver that can export to VTK. For all other solvers, the package supports the import of pathline data from a CSV file. The only requirement is that velocity gradient data is available along the pathlines. Then the functionalities 5 and 6 are available.

I am currently using the package to post-process Eulerian flow data from our in-house finite-element solver :code:`XNS`. I am further planning on extending compatibility to :code:`.vtu` files (COMSOL) and :code:`.cgns` files (ANSYS Fluent, Star-CCM). If you are interested in using the package with a different solver, please let me know and I will try to accommodate your needs.

The package is meant for steady flow data. Rotational flows such as those present in most blood pumps are meant to be handled using the MRF approach to enable a steady solution. Transient flow data needs to be averaged in time to obtain quasi-MRF data before being processed by this package.