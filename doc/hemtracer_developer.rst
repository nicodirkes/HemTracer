.. _developer doc:

Developer documentation
=======================

The developer documentation is organized by modules. This reflects the actual structure of the code. It further lists all private members of classes. This is useful for developers, but not for users.

In contrast, the user documentation directly lists all members of the package that the user can import directly from the hemtracer package. These members are defined in :file:`__init__.py` For example, the user imports the :class:`hemtracer.eulerian_flow_field.EulerianFlowField` class from the :mod:`hemtracer` package as :code:`from hemtracer import EulerianFlowField`, not from the :mod:`hemtracer.eulerian_flow_field` module. This should make working with the package easier for the user.

:mod:`eulerian_flow_field`
-----------------------------------

.. automodule:: hemtracer.eulerian_flow_field
   :members:
   :show-inheritance:
   :undoc-members:
   :private-members:
   :noindex:


:mod:`pathlines`
-----------------------------------

.. automodule:: hemtracer.pathlines
   :members:
   :show-inheritance:
   :undoc-members:
   :private-members:
   :noindex:

:mod:`hemolysis_solver`
-----------------------------------

.. automodule:: hemtracer.hemolysis_solver
   :members:
   :show-inheritance:
   :undoc-members:
   :private-members:
   :noindex:

:mod:`rbc_model`
-----------------------------------

.. automodule:: hemtracer.rbc_model.rbc_model
   :members:
   :inherited-members:
   :private-members:
   :noindex:

.. automodule:: hemtracer.rbc_model.stress_based.stress_based
   :members:
   :show-inheritance:
   :undoc-members:
   :private-members:
   :noindex:

.. automodule:: hemtracer.rbc_model.strain_based.strain_based
   :members:
   :show-inheritance:
   :undoc-members:
   :private-members:
   :noindex:

:mod:`hemolysis_model`
-----------------------------------

.. automodule:: hemtracer.hemolysis_model.hemolysis_model
   :members:
   :show-inheritance:
   :undoc-members:
   :private-members:
   :noindex: