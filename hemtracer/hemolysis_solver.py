from __future__ import annotations
from hemtracer.rbc_model import RBCModel
from hemtracer.hemolysis_model import PowerLawModel
from hemtracer.pathlines import PathlineTracker
from typing import List, Dict
from numpy.typing import ArrayLike

class HemolysisSolver:
    """
    Class for computing hemolysis along pathlines. Takes an existing pathline tracker and handles the various pathlines contained within, interfacing to the hemolysis models.
    """

    _pathline_tracker: PathlineTracker = None  
    """
    Pathline tracker object.
    """

    _v_name: str = None  
    """
    Name of velocity attribute on pathlines.
    """

    _dv_name: str = None  
    """
    Name of velocity gradient attribute on pathlines.
    """

    _omega_name: str = None  
    """
    Name of angular velocity attribute on pathlines.
    """

    _r_name: str = None  
    """
    Name of orthogonal distance to center of rotation attribute on pathlines.
    """

    def __init__(self, pathline_tracker: PathlineTracker) -> None:
        """
        Associate pathline tracker with hemolysis solver and get names of relevant quantities.

        :param pathline_tracker: Pathline tracker object.
        :type pathline_tracker: PathlineTracker
        """
        self._pathline_tracker = pathline_tracker
        self._v_name = self._pathline_tracker.get_name_velocity() # Velocity.
        self._dv_name = self._pathline_tracker.get_name_velocity_gradient() # Velocity gradient.
        self._omega_name = self._pathline_tracker.get_name_omega_frame() # Angular velocity of frame of reference.
        self._r_name = self._pathline_tracker.get_name_distance_center() # Orthogonal distance to center of rotation.

        if self._dv_name is None: 
            raise AttributeError('No velocity gradient data available on pathlines.')


    def compute_representativeShear(self, model: RBCModel) -> None:
        """
        Obtain representative scalar shear rate (effective shear) from stress-based or strain-based cell model.

        :param model: Cell model to use.
        :type model: RBCModel
        """
        i=0
        pathlines = self._pathline_tracker.get_pathlines()
        n_total = len(pathlines)
        G_rep_name = self._get_attribute_name_representativeShear(model)

        print('Integrating ' +  model.get_name() + ' model along pathlines...')
        for pathline in pathlines:

            # Unpack pathline information.
            t0 = pathline.get_t0()
            tend = pathline.get_tend()
            om = pathline.get_attribute_interpolator(self._omega_name)
            dv = pathline.get_attribute_interpolator(self._dv_name)
            r = pathline.get_attribute_interpolator(self._r_name)
            v = pathline.get_attribute_interpolator(self._v_name)

            # Give pathline information to model.
            model.set_time_dependent_quantitites(t0, tend, dv, om, r, v)

            # Solve model.
            (t, G_rep) = model.compute_representative_shear()

            # Store Geff in pathline.
            pathline.add_attribute(t, G_rep, G_rep_name)

            i+=1
            print("...finished " + str(i) + " out of " + str(n_total) + " pathlines.")

    def compute_hemolysis(self, cell_model: RBCModel, powerlaw_model: PowerLawModel) -> None:
        """
        Computes index of hemolysis along pathlines in percent.

        :param cell_model: Model that was used to compute scalar shear rate.
        :type cell_model: RBCModel
        :param powerlaw_model: Power law model to use for computing index of hemolysis.
        :type powerlaw_model: PowerLawModel
        """

        cell_model_solutions = self.get_representativeShear(cell_model)
        pathlines = self._pathline_tracker.get_pathlines()

        n_total = len(cell_model_solutions)
        i=0

        print('Integrating ' + powerlaw_model.get_name() + ' power law along pathlines using ' + cell_model.get_name() + ' cell model...')
        for (sol, pl) in zip(cell_model_solutions, pathlines):
            
            
            t = sol['t']
            G = sol['y']

            IH = powerlaw_model.compute_hemolysis(t, G)
            
            pl.add_attribute(t, IH, self._get_attribute_name_hemolysis(powerlaw_model, cell_model))

            i+=1
            print("...finished " + str(i) + " out of " + str(n_total) + " pathlines.")

    def _get_attribute_name_representativeShear(self, model: RBCModel) -> str:
        """
        Returns name of representative shear rate attribute on pathlines.

        :param model: Cell model to use.
        :type model: RBCModel
        :return: Name of representative shear rate attribute.
        :rtype: str
        """

        representative_shear_rate_name = 'G'
        return representative_shear_rate_name + '_' + model.get_name()
    
    def _get_attribute_name_hemolysis(self, powerlaw_model: PowerLawModel, cell_model: RBCModel) -> str:
        """
        Returns name of hemolysis attribute on pathlines.

        :param powerlaw_model: Power law model to use.
        :type powerlaw_model: PowerLawModel
        :param cell_model: Cell model to use.
        :type cell_model: RBCModel
        :return: Name of hemolysis attribute.
        :rtype: str
        """

        index_of_hemolysis_name = 'IH'
        return index_of_hemolysis_name + '_' + powerlaw_model.get_name() + '_' + cell_model.get_name()
    
    def get_representativeShear(self, model: RBCModel) -> List[Dict[str, ArrayLike]]:
        """
        Obtain representative scalar shear rate along pathlines after it has been computed. Returns a list of dictionaries, each one representing a pathline and containing the keys 't' and 'val' for time and representative shear rate.

        :param model: Cell model to use.
        :type model: RBCModel
        :return: List of dictionaries, each one representing a pathline and containing the keys 't' and 'val' for time and representative shear rate.
        :rtype: List[Dict[str, ArrayLike]]
        """

        return self._pathline_tracker.get_attribute(self._get_attribute_name_representativeShear(model))