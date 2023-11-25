.. _user doc:

User documentation
=====================

.. automodule:: hemtracer
   :members:
   :inherited-members:
   :imported-members:

.. _rbc models api:

RBC models
-----------------

.. automodule:: hemtracer.rbc_model.strain_based
   
   .. autoclass:: hemtracer.rbc_model.strain_based.AroraFullEig
      :members:
      :inherited-members:
   
   .. autoclass:: hemtracer.rbc_model.strain_based.AroraSimplified
      :members:
      :inherited-members:
   
   .. autoclass:: hemtracer.rbc_model.strain_based.TankTreading
      :members:
      :inherited-members:
   
   .. autoclass:: hemtracer.rbc_model.strain_based.TankTreadingRotationCorrection
      :members:
      :inherited-members:
   
   .. autoclass:: hemtracer.rbc_model.strain_based.MorphologyModelCoefficients
      :class-doc-from: class
      :members:

.. automodule:: hemtracer.rbc_model.stress_based

   .. autoclass:: hemtracer.rbc_model.stress_based.Bludszuweit
      :members:
      :inherited-members:
   
   .. autoclass:: hemtracer.rbc_model.stress_based.Frobenius
      :members:
      :inherited-members:

   .. autoclass:: hemtracer.rbc_model.stress_based.SecondInvariant
      :members:
      :inherited-members:

Hemolysis models
-----------------
.. automodule:: hemtracer.hemolysis_model

   .. autoclass:: hemtracer.hemolysis_model.PowerLawModel
      :class-doc-from: both
   
   .. autoclass:: hemtracer.hemolysis_model.IHCorrelation
      :class-doc-from: class
      :members: