from __future__ import annotations
from hemtracer.eulerian_flow_field import EulerianFlowField
from vtk import vtkStreamTracer
from scipy.interpolate import interp1d
import numpy as np
from typing import List, Dict, Any
from numpy.typing import ArrayLike
from hemtracer._definitions import Vector3, vtk_point_assoc
from collections.abc import Callable


integration_time_name = 'IntegrationTime'   # integration time along pathline

class PathlineAttribute:
    """Class for storing a quantity of interest along a pathline. The data is stored as a function of the integration time. All class attributes are meant to be publicly accessible.
    """

    t: ArrayLike = None  
    """
    The integration times along the pathline at which the values are stored.
    """

    val: ArrayLike = None  
    """
    The values of the quantity of interest along the pathline.
    """

    name: str = None  
    """
    The name of the quantity of interest.
    """

    interpolator: Callable[[float], Any] = None  
    """
    A function that interpolates the values along the pathline.
    """

    def __init__(self, t: ArrayLike, val: ArrayLike, name: str, interpolation_scheme: str='linear') -> None:
        """
        :param t: The integration times along the pathline.
        :type t: ArrayLike
        :param val: The values of the attribute along the pathline.
        :type val: ArrayLike
        :param name: The name of the attribute.
        :type name: str
        :param interpolation_scheme: The scheme to use for interpolation, defaults to 'linear'. Refer to scipy.interpolate.interp1d for other options. 
        :type interpolation_scheme: str
        """

        self.t = t
        self.val = val
        self.name = name
        self.interpolator = interp1d(self.t, self.val, axis=0, kind=interpolation_scheme)

class Pathline:
    """
    Class representing a single pathline as a collection of PathlineAttribute objects. Every pathline has an attribute 'Position', which is stored by default as definition of the pathline. Additional attributes can be added using the add_attribute function.
    """

    _attributes: List[PathlineAttribute] = None  
    """
    A list of PathlineAttribute objects.
    """

    _t0: float = None  
    """
    The initial integration time of the pathline.
    """

    _tend: float = None  
    """
    The final integration time of the pathline.
    """

    def __init__(self, t: List[float], x: List[Vector3]) -> None:
        """
        Constructing an object creates its 'Position' attribute and stores it in attributes list.

        :param t: The integration times along the pathline.
        :type t: List[float]
        :param x: The positions along the pathline.
        :type x: List[Vector3]
        """

        if len(t) > 1:
            self._t0 = t[0]
            self._tend = t[-1]
        else:
            raise ValueError("Pathline must have at least two points.")
        
        self._attributes = [ PathlineAttribute(t, x, 'Position') ]

    def get_attribute(self, name: str) -> PathlineAttribute:
        """
        Returns the attribute with the specified name. Returns None if no attribute with the specified name exists.

        :param name: The name of the attribute.
        :type name: str
        :return: The attribute with the specified name or None.
        :rtype: PathlineAttribute
        """
        
        for attribute in self._attributes:
            if attribute.name == name:
                return attribute
        return None
    

    def get_attribute_names(self) -> List[str]:
        """
        Returns a list of all attributes stored in pathline.

        :return: A list of attribute names.
        :rtype: List[str]
        """

        return [ attribute.name for attribute in self._attributes ]
    
    def get_attribute_interpolator(self, attribute_name: str) -> interp1d:
        """
        Returns interpolator for attribute on pathline, or None if it does not exist.
        Arguments:
        attribute_name : string indicating name of attribute
        """

        if attribute_name is None:
            return None
        
        attribute = self.get_attribute(attribute_name)
        if attribute is None:
            return None
        return attribute.interpolator

    def add_attribute(self, t: ArrayLike, val: ArrayLike, name: str, interpolation_scheme: str='linear') -> None:
        """
        Adds an attribute to the pathline.

        :param t: The integration times along the pathline.
        :type t: ArrayLike
        :param val: The values of the attribute along the pathline.
        :type val: ArrayLike
        :param name: The name of the attribute.
        :type name: str
        :param interpolation_scheme: The interpolation scheme to use along the pathline. Defaults to 'linear'.
        :type interpolation_scheme: str
        """

        # Check if attribute with same name already exists and print warning.
        for attribute in self._attributes:
            if attribute.name == name:
                print("Attribute with name " + name + " already exists on pathline.")
                print("Overwriting attribute.")
                self._attributes.remove(attribute)

        self._attributes.append(PathlineAttribute(t, val, name, interpolation_scheme))
    
    def get_t0(self) -> float:
        """
        Returns the initial integration time of the pathline.

        :return: The initial integration time.
        :rtype: float
        """

        return self._t0

    def get_tend(self) -> float:
        """
        Returns the final integration time of the pathline.

        :return: The final integration time.
        :rtype: float
        """

        return self._tend

class PathlineTracker:
    """
    Class for tracking pathlines in a flow field. Uses VTK's vtkStreamTracer to compute pathlines. Stores pathlines as :class:`Pathline` objects. All point-centered data available in the Eulerian field is interpolated to the pathlines. Cell-centered data is not interpolated. Data can be interpolated to the pathlines afterwards using the :func:`interpolate_to_pathlines` function.
    """

    _velocity_name: str = None  
    """
    The name of the velocity field to use for pathline integration.
    """

    _flow_field: EulerianFlowField = None  
    """
    The EulerianFlowField object in which to track pathlines.
    """

    _pathlines: List[Pathline] = []  
    """
    A list of Pathline objects, empty if none have been computed yet.
    """

    def __init__(self, flow_field: EulerianFlowField) -> None:
        """
        Associate Eulerian flow field data and find appropriate velocity field.

        :param flow_field: The flow field in which to track pathlines.
        :type flow_field: EulerianFlowField
        """

        self._flow_field = flow_field

        # Find name of relevant velocity field.
        self._velocity_name = self._flow_field.get_name_advection_velocity()
        
    
    def compute_pathlines(self, x0: List[Vector3], 
                          initial_step: float = 0.001, min_step: float = 0.001, max_step: float = 0.002, 
                          max_err: float = 1e-3, max_length: float = 5.0, n_steps: float = 100000) -> None:
        """
        Compute pathlines starting from a list of initial points. Stores pathlines in internal list. All point-centered data available in the Eulerian field is interpolated to the pathlines. Cell-centered data is not interpolated. Data can be interpolated to the pathlines afterwards using the interpolate_to_pathlines function.

        :param x0: A list of seed points.
        :type x0: List[Vector3]
        :param initial_step: The initial step size for the pathline integration. Defaults to 0.001.
        :type initial_step: float
        :param min_step: The minimum step size for the pathline integration. Defaults to 0.001.
        :type min_step: float
        :param max_step: The maximum step size for the pathline integration. Defaults to 0.002.
        :type max_step: float
        :param max_err: The maximum error for the pathline integration. Defaults to 1e-3.
        :type max_err: float
        :param max_length: The maximum length of the pathline. Defaults to 5.0.
        :type max_length: float
        :param n_steps: The maximum number of steps to take. Defaults to 100000.
        :type n_steps: float
        """

        n_total = len(x0)
        i = 0

        print("Computing pathlines...")
        for x0_i in x0:
            tracer = vtkStreamTracer()
            tracer.SetInputData(self._flow_field.get_vtk_flow_field())
            tracer.SetInputArrayToProcess(0, 0, 0, vtk_point_assoc, self._velocity_name)
            tracer.SetInterpolatorTypeToDataSetPointLocator()
            tracer.SetStartPosition(x0_i)
            tracer.SetIntegrationDirectionToForward()
            tracer.SetMaximumPropagation(max_length)
            tracer.SetMaximumIntegrationStep(max_step)
            tracer.SetInitialIntegrationStep(initial_step)
            tracer.SetMinimumIntegrationStep(min_step)
            tracer.SetIntegratorTypeToRungeKutta45()
            tracer.SetMaximumError(max_err)
            tracer.SetMaximumNumberOfSteps(n_steps)
            tracer.SetComputeVorticity(False)
            tracer.Update()
        
            # Get the values in np format
            points_np = np.array(tracer.GetOutput().GetPoints().GetData(), copy=True)
            point_data = tracer.GetOutput().GetPointData()
            t_np = np.array(point_data.GetArray(integration_time_name), copy=True)

            # Store results in Pathline object.
            pl = Pathline(t_np, points_np)

            # Additionally store interpolated field data and integration time.
            for j in range(point_data.GetNumberOfArrays()):
                name = point_data.GetArrayName(j)
                pl.add_attribute(t_np, np.array(point_data.GetArray(name), copy=True), name)

            self._pathlines.append(pl)
            i = i+1
            print("...finished " + str(i) + " out of " + str(n_total) + " pathlines.")

    def interpolate_dv_to_pathlines(self, sampling_rate: float = 0.001, interpolation_scheme: str = 'previous') -> None:
        """
        Special case of interpolate_to_pathlines: interpolate velocity gradient data to pathlines. If velocity gradient has not been computed yet, it is computed as part of this function.

        :param sampling_rate: The sampling rate for interpolation. Positive values are interpreted as the (fixed) time interval between subsequent samples on the pathline. Zero means to use the same time points as the pathline integration. Negative values are not allowed. Defaults to 0.001.
        :type sampling_rate: float
        :param interpolation_scheme: The interpolation scheme to use along the pathline. For all possible values, refer to scipy.interpolate.interp1d. Defaults to 'previous'.
        :type interpolation_scheme: str
        """

        # Compute velocity gradient if necessary.
        velocity_gradient_name = self._flow_field.get_name_velocity_gradient()
        if velocity_gradient_name is None:
            self._flow_field.compute_gradients()
            velocity_gradient_name = self._flow_field.get_name_velocity_gradient()
        
        # Interpolate velocity gradient.
        self.interpolate_to_pathlines([velocity_gradient_name], 
                                      sampling_rate=sampling_rate, interpolation_scheme=interpolation_scheme)

    def interpolate_to_pathlines(self, field_names: List[str], 
                                 sampling_rate: float = 0.0, interpolation_scheme: str = 'linear') -> None:
        """
        Interpolate field data (cell-centered or point-centered) to pathlines. If fields of same names already exist on pathlines, they are overwritten.

        :param field_names: A list of field names to interpolate.
        :type field_names: List[str]
        :param sampling_rate: The sampling rate for interpolation. Positive values are interpreted as the (fixed) time interval between subsequent samples on the pathline. Zero means to use the same time points as the pathline integration. Defaults to 0.0.
        :type sampling_rate: float
        :param interpolation_scheme: The interpolation scheme to use along the pathline. For all possible values, refer to scipy.interpolate.interp1d. Defaults to 'linear'.
        :type interpolation_scheme: str
        """

        i = 0
        print("Interpolating field data to pathlines...")
        for pathline in self._pathlines:

            # Get integration times and interpolator.
            t = pathline.get_attribute('Position').t
            x_interp = pathline.get_attribute('Position').interpolator

            # Determine time points at which to interpolate.
            if sampling_rate > 0:
                n_samples = int((t[-1]-t[0])/sampling_rate)+1
                ti = np.linspace(t[0], t[-1], n_samples, endpoint=True)
            else:
                ti = t

            # Interpolate.
            xi = x_interp(ti)
            val_interp = self._flow_field.probe_field(xi, field_names)

            for field_name in field_names:
                pathline.add_attribute(ti, val_interp[field_name], 
                                       field_name, interpolation_scheme=interpolation_scheme)
            
            i = i+1
            print("...finished " + str(i) + " out of " + str(len(self._pathlines)) + " pathlines.")

    def get_pathlines(self) -> List[Pathline]:
        """
        Returns the current list of pathlines.

        :return: A list of pathlines.
        :rtype: List[Pathline]
        """

        return self._pathlines
    
    def get_flow_field(self) -> EulerianFlowField:
        """
        Returns the flow field associated with the pathline tracker.

        :return: The flow field.
        :rtype: EulerianFlowField
        """

        return self._flow_field

    def get_attribute(self, attribute_name: str) -> List[Dict[str, ArrayLike]]:
        """
        Returns a list of dictionaries, each one representing a pathline and containing the keys 't' and 'y' for time and attribute, respectively.

        :param attribute_name: The name of the attribute to return.
        :type attribute: str
        :return: A list of dictionaries.
        :rtype: List[Dict[str, ArrayLike]]
        """

        dict_list = []
        for pl in self._pathlines:
            attribute = pl.get_attribute(attribute_name)
            if attribute is None:
                raise AttributeError('No attribute with name ' + attribute_name + ' found on pathline.')
            dict_list.append({'t': attribute.t, 'y': attribute.val})

        return dict_list
    
    def get_name_velocity(self) -> str:
        """
        Returns the name of the velocity field used for pathline integration.

        :return: The name of the velocity field.
        :rtype: str
        """

        return self._velocity_name
    
    def get_name_velocity_gradient(self) -> str:
        """
        Returns the name of the attribute that contains the velocity gradient. Implemented by asking the flow field.

        :return: The name of the attribute.
        :rtype: str
        """

        return self._flow_field.get_name_velocity_gradient()
    
    def get_name_omega_frame(self) -> str:
        """
        Returns the name of the attribute that describes the angular velocity vector of the frame of reference. Implemented by asking the flow field.

        :return: The name of the attribute.
        :rtype: str
        """

        return self._flow_field.get_name_omega_frame()
    
    def get_name_distance_center(self) -> str:
        """
        Returns the name of the attribute that describes the orthogonal distance to the center of rotation. Implemented by asking the flow field.

        :return: The name of the attribute.
        :rtype: str
        """

        return self._flow_field.get_name_distance_center()