from __future__ import annotations
from hemtracer.rbc_model import RBCModel
from hemtracer.hemolysis_model import PowerLawModel
from hemtracer.pathlines import PathlineCollection
from typing import List, Dict
from numpy.typing import NDArray
import numpy as np

class HemolysisSolver:
    """
    Class for computing hemolysis along pathlines. Takes an existing pathline collection and handles the various pathlines contained within, interfacing to the hemolysis models.
    """

    _pathlines: PathlineCollection
    """
    Pathlines.
    """

    _v_name: str | None = None
    """
    Name of velocity attribute on pathlines.
    """

    _dv_name: str
    """
    Name of velocity gradient attribute on pathlines.
    """

    _omega_name: str | None = None  
    """
    Name of angular velocity attribute on pathlines.
    """

    _r_name: str | None = None  
    """
    Name of orthogonal distance to center of rotation attribute on pathlines.
    """

    def __init__(self, pathlines: PathlineCollection) -> None:
        """
        Associate pathlines with hemolysis solver and get names of relevant quantities.

        :param pathlines: Pathlines to analyze.
        :type pathlines: PathlineCollection
        """

        self._pathlines = pathlines
        self._v_name = self._pathlines.get_name_velocity() # Velocity.

        dv_name = self._pathlines.get_name_velocity_gradient() # Velocity gradient.
        if dv_name is None:
            raise AttributeError('No velocity gradient data available on pathlines.')
        else:
            self._dv_name = dv_name

        self._omega_name = self._pathlines.get_name_omega_frame() # Angular velocity of frame of reference.
        self._r_name = self._pathlines.get_name_distance_center() # Orthogonal distance to center of rotation.


    def compute_representativeShear(self, model: RBCModel) -> None:
        """
        Obtain representative scalar shear rate (effective shear) from stress-based or strain-based cell model.

        :param model: Cell model to use.
        :type model: RBCModel
        """

        i=0
        pathlines = self._pathlines.get_pathlines()
        n_total = len(pathlines)
        G_rep_name = model.get_attribute_name()

        print('Integrating ' +  model.get_name() + ' model along pathlines...')
        for pathline in pathlines:

            # Unpack pathline information.
            t0 = pathline.get_t0()
            tend = pathline.get_tend()
            om = pathline.get_attribute_interpolator(self._omega_name)
            dv = pathline.get_attribute_interpolator(self._dv_name)
            r = pathline.get_attribute_interpolator(self._r_name)
            v = pathline.get_attribute_interpolator(self._v_name)

            # create dict of initial attribute values
            init = {}
            for attr_name in pathline.get_attribute_names():
                interp = pathline.get_attribute_interpolator(attr_name)
                if interp is not None: # guaranteed, as we are only calling names that exist
                    init[attr_name] = np.squeeze(interp(t0))

            # Give pathline information to model.
            model.set_time_dependent_quantitites(t0, tend, dv, om, r, v, init)

            # Solve model.
            (t, G_rep) = model.compute_representative_shear()

            # Store Geff in pathline.
            pathline.add_attribute(t, G_rep, G_rep_name)

            i+=1
            print("...finished " + str(i) + " out of " + str(n_total) + " pathlines.", end='\r')


    def compute_hemolysis(self, powerlaw_model: PowerLawModel) -> None:
        """
        Computes index of hemolysis along pathlines in percent.

        :param powerlaw_model: Power law model to use for computing index of hemolysis.
        :type powerlaw_model: PowerLawModel
        """

        cell_model_solutions = self._pathlines.get_attribute(powerlaw_model.get_scalar_shear_name())
        pathlines = self._pathlines.get_pathlines()

        n_total = len(cell_model_solutions)
        i=0

        print('Computing ' + powerlaw_model.get_attribute_name() + ' along pathlines')
        for (sol, pl) in zip(cell_model_solutions, pathlines):
            
            
            t = sol['t']
            G = sol['y']

            IH = powerlaw_model.compute_hemolysis(t, G)
            
            pl.add_attribute(t, IH, powerlaw_model.get_attribute_name())

            i+=1
            print("...finished " + str(i) + " out of " + str(n_total) + " pathlines.", end='\r')


    def get_output(self, model: RBCModel | PowerLawModel) -> List[Dict[str, NDArray]]:
        """
        Obtain hemolysis solutions along pathlines after they have been computed. Returns a list of dictionaries, each one representing a pathline and containing the keys 't' and 'y' for time and output variable.

        :param model: Model to consider.
        :type model: str
        :return: List of dictionaries, each one representing a pathline and containing the keys 't' and 'y' for time and output variable.
        :rtype: List[Dict[str, NDArray]]
        """

        return self._pathlines.get_attribute(model.get_attribute_name())
    
    def average_hemolysis(self, model: PowerLawModel) -> float:
        """
        Average hemolysis index over the end points of all pathlines.

        :param model: Power law model to use.
        :type model: PowerLawModel
        :return: Average hemolysis index.
        :rtype: float
        """

        IHs = self._pathlines.get_attribute(model.get_attribute_name())
        IHs_end = [IH['y'][-1] for IH in IHs]

        return float(np.mean(IHs_end))