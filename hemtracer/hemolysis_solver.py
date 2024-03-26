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

    def __init__(self, pathlines: PathlineCollection) -> None:
        """
        Associate pathlines with hemolysis solver and get names of relevant quantities.

        :param pathlines: Pathlines to analyze.
        :type pathlines: PathlineCollection
        """

        # Initialize attributes.
        self._pathlines = pathlines # Pathlines.
        self._v_name = self._pathlines.get_name_velocity() # Velocity.

        dv_name = self._pathlines.get_name_velocity_gradient() # Velocity gradient.
        if dv_name is None:
            raise AttributeError('No velocity gradient data available on pathlines.')
        else:
            self._dv_name = dv_name

        self._omega_name = self._pathlines.get_name_omega_frame() # Angular velocity of frame of reference.
        self._r_name = self._pathlines.get_name_distance_center() # Orthogonal distance to center of rotation.


    def compute_representativeShear(self, model: RBCModel, store_solution: bool = False) -> None:
        """
        Obtain representative scalar shear rate (effective shear) from stress-based or strain-based cell model.

        :param model: Cell model to use.
        :type model: RBCModel
        :param store_solution: Store solution in pathlines. Defaults to False.
        :type store_solution: bool
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

            # Use position attribute as reference for time points.
            ti = pathline.get_position_attribute().t

            # create dict of initial attribute values
            init = {}
            for attr_name in pathline.get_attribute_names():
                interp = pathline.get_attribute_interpolator(attr_name)
                if interp is not None: # guaranteed, as we are only calling names that exist
                    init[attr_name] = np.squeeze(interp(t0))

            # Give pathline information to model.
            model.set_time_dependent_quantitites(t0, tend, ti, dv, om, r, v, init)

            # Solve model.
            (t, G_rep, sol) = model.compute_representative_shear()

            # Store Geff in pathline.
            pathline.add_attribute(t, G_rep, G_rep_name)

            # Store solution if requested.
            if store_solution:
                if sol is None:
                    raise AttributeError('No solution available for ' + model.get_name() + ' model.')
                for i in range(sol.shape[1]):
                    pathline.add_attribute(t, sol[:,i], model.get_name() + '_sol_' + str(i))

            i+=1
            print("...finished " + str(i) + " out of " + str(n_total) + " pathlines.", end='\r')


    def compute_hemolysis(self, powerlaw_model: PowerLawModel, visc: float|str = 0.0035) -> None:
        """
        Computes index of hemolysis along pathlines in percent.

        :param powerlaw_model: Power law model to use for computing index of hemolysis.
        :type powerlaw_model: PowerLawModel
        :param visc: Blood viscosity. Defaults to 0.0035 Pa s. If a string is given, it is assumed to be the name of an attribute containing the (local) dynamic viscosity.
        :type visc: float | str
        """

        cell_model_solutions = self._pathlines.get_attribute(powerlaw_model.get_scalar_shear_name())
        pathlines = self._pathlines.get_pathlines()

        n_total = len(cell_model_solutions)
        i=0

        print('Computing ' + powerlaw_model.get_attribute_name() + ' along pathlines')
        for (sol, pl) in zip(cell_model_solutions, pathlines):
            
            t = sol['t']
            G = sol['y']

            if isinstance(visc, str):
                mu_interp = pl.get_attribute_interpolator(visc)
                if mu_interp is None:
                    raise AttributeError('No attribute named ' + visc + ' found on pathlines.')
                else:
                    mu = np.squeeze(mu_interp(t))
            else:
                mu = visc * np.ones_like(t)

            IH = powerlaw_model.compute_hemolysis(t, G, mu)
            
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
    
    def average_hemolysis(self, model: PowerLawModel, end_criterion_attribute: str | None = None, end_criterion_value: float = 0.0) -> float:
        """
        Average hemolysis index over the end points of all pathlines.

        :param model: Power law model to use.
        :type model: PowerLawModel
        :param end_criterion_name: Criterion for selecting end points. Can be a string representing an attribute name. If given, uses the first time point where the attribute is greater than end_criterion_value as end point. Discards all pathlines where the criterion is not met. Defaults to None.
        :type end_criterion: str | None
        :param end_criterion_value: Value to use as end criterion. Defaults to 0.0.
        :type end_criterion_value: float
        :return: Average hemolysis index.
        :rtype: float
        """

        IHs_end = []

        # if end_criterion_attribute is given, use it to select end points
        if end_criterion_attribute is not None:

            # get end criterion values for all pathlines
            end_criterion_list = self._pathlines.get_attribute(end_criterion_attribute)
            if end_criterion_list is None:
                raise AttributeError('No attribute named ' + end_criterion_attribute + ' found on pathlines.')
            
            # loop over pathlines
            for (i_pl,end_criterion_pl) in enumerate(end_criterion_list):

                # find index of first time point where end_criterion is greater than zero
                end_criterion = end_criterion_pl['y']
                t = end_criterion_pl['t']
                idx = np.argmax(end_criterion > end_criterion_value)

                # get IH at this time point
                t_end = t[idx]
                IH_pl = self._pathlines.get_pathlines()[i_pl].get_attribute_interpolator(model.get_attribute_name())
                if IH_pl is None:
                    raise AttributeError('No attribute named ' + model.get_attribute_name() + ' found on pathlines.')
                IHs_end.append(IH_pl(t_end))

        # otherwise, use the last time point for each pathline
        else:
            IHs = self._pathlines.get_attribute(model.get_attribute_name())
            IHs_end = [IH['y'][-1] for IH in IHs]

        return float(np.mean(IHs_end))