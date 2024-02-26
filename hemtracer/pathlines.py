from __future__ import annotations
import csv
from hemtracer.eulerian_flow_field import EulerianFlowField
from vtk import vtkStreamTracer, vtkDoubleArray, vtkPolyData, vtkPolyDataWriter, vtkPoints, vtkIntArray, vtkCellArray, vtkPolyLine
from vtkmodules.util.numpy_support import numpy_to_vtk
from scipy.interpolate import interp1d
import numpy as np
from typing import List, Dict, Any, Tuple
from numpy.typing import ArrayLike, NDArray
from hemtracer._definitions import Vector3, vtk_point_assoc
import pandas as pd


integration_time_name = 'IntegrationTime'   # integration time along pathline
position_name = 'Position'                  # position along pathline

class PathlineAttribute:
    """Class for storing a quantity of interest along a pathline. The data is stored as a function of the integration time. All class attributes are meant to be publicly accessible.
    """

    t: NDArray
    """
    The integration times along the pathline at which the values are stored.
    """

    val: NDArray
    """
    The values of the quantity of interest along the pathline.
    """

    name: str
    """
    The name of the quantity of interest.
    """

    interpolator: interp1d
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

        self.t = np.asarray(t)
        self.val = np.asarray(val)
        self.name = name
        self.interpolator = interp1d(self.t, self.val, axis=0, kind=interpolation_scheme, assume_sorted=True)

    def get_number_of_components(self) -> int:
        """
        Returns the number of components of the attribute.

        :return: The number of components.
        :rtype: int
        """

        if self.val.ndim == 1:
            return 1

        return self.val.shape[1]

class Pathline:
    """
    Class representing a single pathline as a collection of PathlineAttribute objects. Every pathline has an attribute 'Position', which is stored by default as definition of the pathline. Additional attributes can be added using the add_attribute function.
    """

    _attributes: List[PathlineAttribute]
    """
    A list of PathlineAttribute objects.
    """

    _t0: float
    """
    The initial integration time of the pathline.
    """

    _tend: float
    """
    The final integration time of the pathline.
    """

    _reason_for_termination: str
    """
    The reason for termination of the pathline integration. Can be 'out of domain', 'not initialized', 'unexpected value', 'out of length', 'out of steps', 'stagnation', 'misc'.
    """

    def __init__(self, t: List[float], x: List[Vector3], reason_for_termination: str = 'misc') -> None:
        """
        Constructing an object creates its 'Position' attribute and stores it in attributes list.

        :param t: The integration times along the pathline.
        :type t: List[float]
        :param x: The positions along the pathline.
        :type x: List[Vector3]
        :param reason_for_termination: The reason for termination of the pathline integration, defaults to 'misc'.
        :type reason_for_termination: str
        """

        if len(t) == 1:
            raise ValueError("Pathline must have at least two points.")
        
        self._t0 = t[0]
        self._tend = t[-1]
        self._attributes = [ PathlineAttribute(t, x, position_name) ]
        self._reason_for_termination = reason_for_termination

    def get_position_attribute(self) -> PathlineAttribute:
        """
        Returns the 'Position' attribute of the pathline. Has to exist by definition.

        :raises AttributeError: If no position attribute exists on pathline.
        :return: The 'Position' attribute.
        :rtype: PathlineAttribute
        """

        position_attribute = self.get_attribute(position_name)
        if position_attribute is None:
            raise AttributeError('No position attribute found on pathline.')
        
        return position_attribute
    
    def unify_attributes(self, attribute_names: List[str] | None, ref_attribute_name: str = position_name) -> Tuple[NDArray, List[str]]:
        """
        Interpolates all attributes to the same time points of some reference attribute (if not specified, use the 'Position' attribute) and return them as an array. The first column of each array contains the time points used.

        :param attribute_names: A list of attribute names to unify. If None, all attributes are unified.
        :type attribute_names: List[str] | None
        :param ref_attribute_name: The name of the attribute to use as reference, defaults to 'Position'.
        :type ref_attribute_name: str
        :return: A tuple containing the array of interpolated values and a list of attribute names. The first column of the array contains the time points used.
        :rtype: Tuple[NDArray, List[str]]
        """

        # Get integration times of reference attribute.
        ref_attribute = self.get_attribute(ref_attribute_name)
        if ref_attribute is None:
            raise AttributeError('Reference attribute ' + ref_attribute_name + ' not found on pathline.')
        t_ref = ref_attribute.t
        
        # Find attributes to unify.
        if attribute_names is None:
            attribute_names = self.get_attribute_names()
        
        # Determine the total number of components.
        n_components_total = 1 # for time
        for attribute_name in attribute_names:
            attribute = self.get_attribute(attribute_name)
            if attribute is None:
                raise AttributeError('No attribute with name ' + attribute_name + ' found on pathline.')
            n_components_total += attribute.get_number_of_components()

        # Initialize output array for interpolated values.
        vals_interp = np.zeros((len(t_ref), n_components_total))
        
        # Interpolate attributes.
        i_c = 1 # index for total component, start at 1 because first column is time
        attribute_names_comp = [] # list of all attribute names, including components
        for attribute_name in attribute_names:
            attribute = self.get_attribute(attribute_name)
            if attribute is None:
                raise AttributeError('No attribute with name ' + attribute_name + ' found on pathline.')
            n_c = attribute.get_number_of_components()
            
            # Add attribute names to list. If attribute is vector-valued, add component index.
            if n_c > 1:
                attribute_names_comp.extend([attribute_name + '_' + str(i) for i in range(n_c)])
            else:
                attribute_names_comp.append(attribute_name)

            # Add (interpolated) attribute values to array.
            i_c_new = i_c + n_c
            interpolated_attribute = attribute.interpolator(t_ref)
            vals_interp[:,i_c:i_c_new] = interpolated_attribute
            i_c = i_c_new
        
        # Add integration times to array.
        vals_interp[:,0] = t_ref
        attribute_names_comp.insert(0, integration_time_name)

        return (vals_interp, attribute_names_comp)


    def get_attribute(self, name: str) -> PathlineAttribute|None:
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
    
    def get_attribute_interpolator(self, attribute_name: str|None) -> interp1d|None:
        """
        Returns interpolator for attribute on pathline, or None if it does not exist.

        :param attribute_name: string indicating name of attribute
        :type attribute_name: str|None
        :return: interpolator for attribute on pathline, or None if it does not exist
        :rtype: interp1d|None
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

        :param t: The integration times along the pathline. If not in the range [t0, tend], the integration times are clipped.
        :type t: ArrayLike
        :param val: The values of the attribute along the pathline (first axis must correspond to integration times). If the attribute is vector-valued, the second axis must correspond to the components of the vector.
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
        
        t_att = np.squeeze(np.asarray(t))
        val_att = np.asarray(val)
        if val_att.ndim == 1: # ensure 2D array
            val_att = val_att[:,np.newaxis]

        if len(t_att) != len(val_att):
            raise ValueError("Length of integration times and values must be equal.")
        
        if len(t_att) == 1:
            raise ValueError("Pathline must have at least two points.")

        # If times are not in range, clip integration times and interpolate values to new bounds.
        if t_att[0] < self._t0 or t_att[-1] > self._tend:
            t_clip = np.clip(t_att, self._t0, self._tend)
            val_interp = interp1d(t, val, axis=0, kind=interpolation_scheme, assume_sorted=True)
            val_att = np.asarray(val_interp(t_clip))
            t_att = t_clip

        # If necessary, extrapolate values to initial and final integration time by repeating first and last value.
        if t_att[0] > self._t0:
            t_att = np.insert(t_att, 0, self._t0)
            val_att = np.insert(val_att, 0, val_att[0,:], axis=0)
        if t_att[-1] < self._tend:
            t_att = np.append(t_att, self._tend)
            val_att = np.append(val_att, val_att[-1:,:], axis=0)

        self._attributes.append(PathlineAttribute(t_att, val_att, name, interpolation_scheme))
    
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
    
    def get_reason_for_termination(self) -> str:
        """
        Returns the reason for termination of the pathline integration.

        :return: The reason for termination.
        :rtype: str
        """

        return self._reason_for_termination

class PathlineCollection:
    """
    Class for storing a collection of pathlines. 
    """

    _pathlines: List[Pathline] = []
    """
    A list of Pathline objects.
    """

    _velocity_name: str | None = None
    """
    The name of the velocity field.
    """

    _velocity_gradient_name: str | None = None
    """
    The name of the velocity gradient field.
    """

    _omega_frame_name: str | None = None
    """
    The name of the angular velocity vector of the frame of reference.
    """

    _distance_center_name: str | None = None
    """
    The name of the field for the orthogonal distance to the center of rotation.
    """

    def get_pathlines(self) -> List[Pathline]:
        """
        Returns the current list of pathlines.

        :return: A list of pathlines.
        :rtype: List[Pathline]
        """

        return self._pathlines
    
    def get_attribute(self, attribute_name: str) -> List[Dict[str, NDArray]]:
        """
        Returns a list of dictionaries, each one representing a pathline and containing the keys 't' and 'y' for time and attribute, respectively.

        :param attribute_name: The name of the attribute to return.
        :type attribute: str
        :return: A list of dictionaries.
        :rtype: List[Dict[str, NDArray]]
        """

        dict_list = []
        for pl in self._pathlines:
            attribute = pl.get_attribute(attribute_name)
            if attribute is None:
                raise AttributeError('No attribute with name ' + attribute_name + ' found on pathline.')
            dict_list.append({'t': attribute.t, 'y': attribute.val})

        return dict_list
    
    def get_name_velocity(self) -> str | None:
        """
        Returns the name of the velocity field used for pathline integration.

        :return: The name of the velocity field.
        :rtype: str
        """

        return self._velocity_name

    def get_name_velocity_gradient(self) -> str | None:
        """
        Returns the name of the attribute that contains the velocity gradient. None if not available.

        :return: The name of the attribute.
        :rtype: str | None
        """

        return self._velocity_gradient_name
    
    def get_name_omega_frame(self) -> str | None:
        """
        Returns the name of the attribute that describes the angular velocity vector of the frame of reference. None if not available.

        :return: The name of the attribute.
        :rtype: str | None
        """

        return self._omega_frame_name
    
    def get_name_distance_center(self) -> str | None:
        """
        Returns the name of the attribute that describes the orthogonal distance to the center of rotation. None if not available.

        :return: The name of the attribute.
        :rtype: str | None
        """

        return self._distance_center_name
    
    def to_file(self, filename: str, attribute_names: List[str] | None = None) -> None:
        """
        Write pathlines to file. If attribute names are specified, only those attributes are written to file. Otherwise, all attributes are written to file.

        Interpolates all attributes to the same time points (those of the 'Position' attribute) before writing to file.

        Supports writing to CSV, npz, and VTK files. In case of CSV and VTK, it is assumed that all pathlines contain the same attributes, as this is required by the file format (every point must have the same number of attributes). If you want to write pathlines with different attributes to file, there are two options: (1) Write to npz file, which does not require all pathlines to have the same attributes. (2) Instantiate multiple PathlineTracker objects, each with a different set of attributes, and write each one to a separate file.
        In case of npz, the attribute names and values on each pathline are stored as separate arrays in the file, pattern: arr_0 = attribute_values_pl1, arr_1 = attribute_names_pl1, arr_2 = attribute_values_pl2, arr_3 = attribute_names_pl2, ....

        :param filename: The name of the file to write to, including path and extension. Supported extensions are .csv, .npz, and .vtk.
        :type filename: str
        :param attribute_names: A list of attribute names to write to file. If None, all attributes are written to file (default).
        :type attribute_names: List[str], optional
        """
        
        print('Writing pathlines to file ' + filename + '...')
        
        pathlines = [ pathline.unify_attributes(attribute_names) for pathline in self._pathlines ]

        # Determine appropriate writer.
        ext = filename.split('.')[-1]
        match ext:
            case 'csv':
                writer = self._write_csv
            case 'npz':
                writer = self._write_npz
            case 'vtk':
                writer = self._write_vtk
            case _:
                raise ValueError('Unsupported file extension. Supported extensions are .csv, .npz, and .vtk.')
        
        # Write to file.
        writer(filename, pathlines)

    def _write_csv(self, filename: str, pathlines: List[Tuple[NDArray,List[str]]]) -> None:
        """
        Write pathlines to CSV file. It is assumed that all pathlines contain the same attributes.

        :param filename: The name of the file to write to, including path and extension.
        :type filename: str
        :param pathlines: A list of tuples, each containing the array of interpolated values and a list of attribute names of a pathline. The first column of the array contains the time points used.
        :type pathlines: List[Tuple[NDArray,List[str]]]
        """
        
        # Store to check if all pathlines contain same attributes.
        attribute_names_ref = pathlines[0][1]

        # Write to file.
        with open(filename, 'w') as f:
            
            # Write header.
            f.write('"PathlineID"')
            f.write('"t"') # time column
            for attribute_name in attribute_names_ref:
                f.write(',"' + attribute_name + '"')
            f.write('\n')

            # Write data.
            for pathline in pathlines:
                if pathline[1] != attribute_names_ref:
                    raise ValueError('Not all pathlines contain the same attributes. Cannot write to CSV file.')
                
                vals = pathline[0]
                for i in range(vals.shape[0]):
                    f.write(str(i))
                    f.write(str(vals[i,0]))
                    for j in range(1,vals.shape[1]):
                        f.write(',' + str(vals[i,j]))
                    f.write('\n')
    
    def _write_npz(self, filename: str, pathlines: List[Tuple[NDArray,List[str]]]) -> None:
        """
        Write pathlines to npz file. It is not assumed that all pathlines contain the same attributes. The attribute names and values on each pathline are stored as separate arrays in the file, pattern: arr_0 = attribute_values_pl1, arr_1 = attribute_names_pl1, arr_2 = attribute_values_pl2, arr_3 = attribute_names_pl2, ....

        :param filename: The name of the file to write to.
        :type filename: str
        :param pathlines: A list of tuples, each containing the array of interpolated values and a list of attribute names of a pathline. The first column of the array contains the time points used.
        :type pathlines: List[Tuple[NDArray,List[str]]]
        """

        # Create interleaved list of attribute values and names.
        attribute_value_name_list = [val for pair in pathlines for val in pair]

        # Write to file.
        np.savez(filename, *attribute_value_name_list)

    def _write_vtk(self, filename: str, pathlines: List[Tuple[NDArray,List[str]]]) -> None:
        """
        Write pathlines to VTK file. All pathlines are written to the same VTK file. They are identified by the attribute 'PathlineID', which is stored as a point data array. 

        :param filename: The name of the file to write to.
        :type filename: str
        :param pathlines: A list of tuples, each containing the array of interpolated values and a list of attribute names of a pathline. The first column of the array contains the time points used.
        :type pathlines: List[Tuple[NDArray,List[str]]]
        """

        # Create a cell array to store the lines in and add the lines to it
        cells = vtkCellArray()

        # Create a vtkPoints object and store the points in it
        points = vtkPoints()

        # Create a polydata to store everything
        polyData = vtkPolyData()

        idx = 0

        for pl_attribute in pathlines:

            # Get pathline attributes.
            x = pl_attribute[0][:,1:4]
            
            # Create points.
            for point in x:
                points.InsertNextPoint(point)
            
            # Create pathline.
            n = len(x)
            polyLine = vtkPolyLine()
            polyLine.GetPointIds().SetNumberOfIds(n)

            for (i, j) in enumerate(np.arange(idx, idx+n)):
                polyLine.GetPointIds().SetId(i, j)
            
            cells.InsertNextCell(polyLine)
            idx = idx+n
        
        # Add everything to the dataset
        polyData.SetPoints(points)
        polyData.SetLines(cells)

        # Add pathline ID as point data array.
        pathline_id_array = vtkIntArray()
        pathline_id_array.SetNumberOfComponents(0)
        pathline_id_array.SetName('PathlineID')
        for i in range(len(pathlines)):
            pathline_id_array.InsertNextValue(i)
        polyData.GetCellData().AddArray(pathline_id_array)

        # Add attributes as point data arrays.
        attribute_names = pathlines[0][1] # assume all pathlines have same attributes
      
        # Aggregate attribute values into array by component.
        attribute_values = pathlines[0][0]
        for pl_attributes in pathlines[1:]:
            if pl_attributes[1] != attribute_names:
                raise ValueError('Not all pathlines contain the same attributes. Cannot write to VTK file.')
            attribute_values = np.append(attribute_values, pl_attributes[0], axis=0)
        
        # Add attribute values to polydata.
        for (i, attribute_name) in enumerate(attribute_names):
            attribute_array = numpy_to_vtk(attribute_values[:,i])
            attribute_array.SetNumberOfComponents(0)
            attribute_array.SetName(attribute_name)
            polyData.GetPointData().AddArray(attribute_array)

        # Write to file
        writer = vtkPolyDataWriter()
        writer.SetInputData(polyData)
        writer.SetFileName(filename)
        writer.Write()

class PathlineReader (PathlineCollection):
    """
    Class for reading pathlines from file. Currently only supports reading from csv files.
    """

    def __init__(self, filename: str, id_name: str, t_name: str,
                    posX_name: str, posY_name: str, posZ_name: str,
                    velX_name:   str | None = None, velY_name:   str | None = None, velZ_name:   str | None = None,
                    dvX_dx_name: str | None = None, dvX_dy_name: str | None = None, dvX_dz_name: str | None = None,
                    dvY_dx_name: str | None = None, dvY_dy_name: str | None = None, dvY_dz_name: str | None = None,
                    dvZ_dx_name: str | None = None, dvZ_dy_name: str | None = None, dvZ_dz_name: str | None = None,
                    omegaX_name: str | None = None, omegaY_name: str | None = None, omegaZ_name: str | None = None,
                    distance_center_name: str | None = None, idx: List[int] | None = None) -> None:
        """
        :param filename: The name of the file to read from.
        :type filename: str
        :param id_name: The name of the attribute containing the pathline IDs.
        :type id_name: str
        :param t_name: The name of the attribute containing the integration times.
        :type t_name: str
        :param posX_name: The name of the attribute containing the x-coordinates of the pathlines.
        :type posX_name: str
        :param posY_name: The name of the attribute containing the y-coordinates of the pathlines.
        :type posY_name: str
        :param posZ_name: The name of the attribute containing the z-coordinates of the pathlines.
        :type posZ_name: str
        :param velX_name: The name of the attribute containing the x-component of the velocity field. Defaults to None.
        :type velX_name: str | None
        :param velY_name: The name of the attribute containing the y-component of the velocity field. Defaults to None.
        :type velY_name: str | None
        :param velZ_name: The name of the attribute containing the z-component of the velocity field. Defaults to None.
        :type velZ_name: str | None
        :param dvX_dx_name: The name of the attribute containing the derivative of v_x w.r.t. x. Defaults to None.
        :type dvX_dx_name: str | None
        :param dvX_dy_name: The name of the attribute containing the derivative of v_x w.r.t. y. Defaults to None.
        :type dvX_dy_name: str | None
        :param dvX_dz_name: The name of the attribute containing the derivative of v_x w.r.t. z. Defaults to None.
        :type dvX_dz_name: str | None
        :param dvY_dx_name: The name of the attribute containing the derivative of v_y w.r.t. x. Defaults to None.
        :type dvY_dx_name: str | None
        :param dvY_dy_name: The name of the attribute containing the derivative of v_y w.r.t. y. Defaults to None.
        :type dvY_dy_name: str | None
        :param dvY_dz_name: The name of the attribute containing the derivative of v_y w.r.t. z. Defaults to None.
        :type dvY_dz_name: str | None
        :param dvZ_dx_name: The name of the attribute containing the derivative of v_z w.r.t. x. Defaults to None.
        :type dvZ_dx_name: str | None
        :param dvZ_dy_name: The name of the attribute containing the derivative of v_z w.r.t. y. Defaults to None.
        :type dvZ_dy_name: str | None
        :param dvZ_dz_name: The name of the attribute containing the derivative of v_z w.r.t. z. Defaults to None.
        :type dvZ_dz_name: str | None
        :param omegaX_name: The name of the attribute containing the x-component of the angular velocity vector of the frame of reference. Defaults to None.
        :type omegaX_name: str | None
        :param omegaY_name: The name of the attribute containing the y-component of the angular velocity vector of the frame of reference. Defaults to None.
        :type omegaY_name: str | None
        :param omegaZ_name: The name of the attribute containing the z-component of the angular velocity vector of the frame of reference. Defaults to None.
        :type omegaZ_name: str | None
        :param distance_center_name: The name of the attribute containing the orthogonal distance to the center of rotation. Defaults to None.
        :type distance_center_name: str | None
        :param idx: A list of indices indicating which pathlines to read from file. If None, all pathlines are read. Defaults to None.
        :type idx: List[int] | None
        """

        print('Reading pathlines from file ' + filename + '...')

        # set attribute names
        pos_names = [posX_name, posY_name, posZ_name]

        self._distance_center_name = distance_center_name

        omega_names = [omegaX_name, omegaY_name, omegaZ_name]
        if any(omega_names):
            self._omega_frame_name = 'OmegaFrame'
        
        vel_names = [velX_name, velY_name, velZ_name]
        if any(vel_names):
            self._velocity_name = 'Velocity'

        vel_grad_names = [dvX_dx_name, dvX_dy_name, dvX_dz_name, dvY_dx_name, dvY_dy_name, dvY_dz_name, dvZ_dx_name, dvZ_dy_name, dvZ_dz_name]
        if any(vel_grad_names):
            self._velocity_gradient_name = 'VelocityGradient'

        # Determine appropriate reader.
        ext = filename.split('.')[-1]
        match ext:
            case 'csv':
                reader = self._read_csv
            case _:
                raise ValueError('Unsupported file extension. Supported extensions are .csv')
        
        # Read from file.
        reader(filename, id_name, t_name, pos_names, vel_names, vel_grad_names, omega_names, distance_center_name, idx)
    
    def _read_csv(self, filename: str, id_name: str, t_name: str, 
                    pos_names: List[str], vel_names: List[str|None], vel_grad_names: List[str|None], omega_names: List[str|None], distance_center_name: str | None, idx: List[int] | None) -> None:
        """
        Read pathlines from CSV file.

        :param filename: The name of the file to read from.
        :type filename: str
        :param id_name: The name of the attribute containing the pathline IDs.
        :type id_name: str
        :param t_name: The name of the attribute containing the integration times.
        :type t_name: str
        :param pos_names: A list of attribute names containing the x, y, and z-coordinates of the pathlines.
        :type pos_names: List[str]
        :param vel_names: A list of attribute names containing the x, y, and z-components of the velocity field. If None, no velocity field is read.
        :type vel_names: List[str|None]
        :param vel_grad_names: A list of attribute names containing the derivatives of the velocity field. If None, no velocity gradient field is read.
        :type vel_grad_names: List[str|None]
        :param omega_names: A list of attribute names containing the components of the angular velocity vector of the frame of reference. If None, no angular velocity vector is read.
        :type omega_names: List[str|None]
        :param distance_center_name: The name of the attribute containing the orthogonal distance to the center of rotation. If None, no distance to center is read.
        :type distance_center_name: str | None
        :param idx: A list of indices indicating which pathlines to read from file. If None, all pathlines are read.
        :type idx: List[int] | None
        """

        # Read from file.
        pl_data = pd.read_csv(filename)

        # Find unique pathline IDs.
        pathline_ids = pl_data[id_name].unique()

        # If indices are specified, only read those pathlines.
        if idx is not None:
            pathline_ids = pathline_ids[idx]

        # Read data.
        for pathline_id in pathline_ids:

            # Get data for pathline.
            pl_data_id = pl_data[pl_data[id_name] == pathline_id]
            t = list(pl_data_id[t_name])
            zeros = np.zeros(len(t))
            x = list(np.asarray([ pl_data_id[name] if name else zeros for name in pos_names ]).T)

            pl = Pathline(t, x)

            # Add additional attributes. If attribute name is None, consider it as zero.
            if self._velocity_name:
                v = np.asarray([ pl_data_id[name] if name else zeros for name in vel_names ]).T
                pl.add_attribute(t, v, self._velocity_name)
            
            if self._velocity_gradient_name:
                dv = np.asarray([ pl_data_id[name] if name else zeros for name in vel_grad_names ]).T
                pl.add_attribute(t, dv, self._velocity_gradient_name)
            
            if self._omega_frame_name:
                omega = np.asarray([ pl_data_id[name] if name else zeros for name in omega_names ]).T
                pl.add_attribute(t, omega, self._omega_frame_name)
            
            if self._distance_center_name:
                d = pl_data_id[distance_center_name] if distance_center_name else zeros
                pl.add_attribute(t, d, self._distance_center_name)

            # Add pathline to collection.
            self._pathlines.append(pl)


class PathlineTracker (PathlineCollection):
    """
    Class for generating pathlines from a given flow field. Uses VTK's vtkStreamTracer to compute pathlines. Stores pathlines as :class:`Pathline` objects. All point-centered data available in the Eulerian field is interpolated to the pathlines. Cell-centered data is not interpolated. Data can be interpolated to the pathlines afterwards using the :func:`interpolate_to_pathlines` function.
    """

    _flow_field: EulerianFlowField
    """
    The EulerianFlowField object in which to track pathlines.
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
                          max_err: float = 1e-3, max_length: float = 5.0, n_steps: int = 100000, terminal_velocity: float = 1e-10, 
                          integrator: str = "RK45", direction: str = "forward") -> None:
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
        :param terminal_velocity: The velocity at which to stop integration. Defaults to 1e-10.
        :type terminal_velocity: float
        :param integrator: The integrator to use. Options are "RK2", "RK4" and "RK45".  Defaults to "RK45".
        :type integrator: str
        :param direction: The direction of integration. Options are "forward", "backward" and "both". Defaults to "forward".
        :type direction: str
        """

        # Check if velocity field is available.
        if self._velocity_name is None:
            raise AttributeError("No velocity field found in flow field. Cannot compute pathlines.")

        n_total = len(x0)
        i = 0

        print("Computing pathlines...")
        for x0_i in x0:
            tracer = vtkStreamTracer()
            tracer.SetInputData(self._flow_field.get_vtk_flow_field())
            tracer.SetInputArrayToProcess(0, 0, 0, vtk_point_assoc, self._velocity_name)
            tracer.SetInterpolatorTypeToDataSetPointLocator()
            tracer.SetStartPosition(tuple(x0_i))
            tracer.SetIntegrationDirectionToForward()
            tracer.SetMaximumPropagation(max_length)
            tracer.SetMaximumIntegrationStep(max_step)
            tracer.SetInitialIntegrationStep(initial_step)
            tracer.SetIntegrationStepUnit(vtkStreamTracer.CELL_LENGTH_UNIT)
            tracer.SetMinimumIntegrationStep(min_step)
            tracer.SetMaximumError(max_err)
            tracer.SetMaximumNumberOfSteps(n_steps)
            tracer.SetTerminalSpeed(terminal_velocity)
            tracer.SetComputeVorticity(False)

            match integrator:
                case "RK2":
                    tracer.SetIntegratorTypeToRungeKutta2()
                case "RK4":
                    tracer.SetIntegratorTypeToRungeKutta4()
                case "RK45":
                    tracer.SetIntegratorTypeToRungeKutta45()
                
            match direction:
                case "forward":
                    tracer.SetIntegrationDirectionToForward()
                case "backward":
                    tracer.SetIntegrationDirectionToBackward()
                case "both":
                    tracer.SetIntegrationDirectionToBoth()
                case _:
                    raise ValueError("Unknown direction " + direction + ". Options are 'forward', 'backward', and 'both'.")

            tracer.Update()
        
            # Get the values in np format
            points_np = np.array(tracer.GetOutput().GetPoints().GetData(), copy=True)
            point_data = tracer.GetOutput().GetPointData()
            t_np = np.array(point_data.GetArray(integration_time_name), copy=True)

            # Ensure correct order of points and integration times.
            idx = np.argsort(t_np)
            t_np = t_np[idx]
            points_np = points_np[idx,:] # sort points accordingly

            # Find reason for termination.
            reason_array = tracer.GetOutput().GetCellData().GetArray('ReasonForTermination')
            if reason_array is None:
                reason_for_termination = 'misc'
            else:
                reason_id = reason_array.GetValue(0)
                match reason_id:
                    case vtkStreamTracer.OUT_OF_DOMAIN:
                        reason_for_termination = 'out of domain'
                    case vtkStreamTracer.NOT_INITIALIZED:
                        reason_for_termination = 'not initialized'
                    case vtkStreamTracer.UNEXPECTED_VALUE:
                        reason_for_termination = 'unexpected value'
                    case vtkStreamTracer.OUT_OF_LENGTH:
                        reason_for_termination = 'out of length'
                    case vtkStreamTracer.OUT_OF_STEPS:
                        reason_for_termination = 'out of steps'
                    case vtkStreamTracer.STAGNATION:
                        reason_for_termination = 'stagnation'
                    case _:
                        reason_for_termination = 'misc'

            # Store results in Pathline object.
            pl = Pathline(list(t_np), list(points_np), reason_for_termination)

            # Additionally store interpolated field data.
            for j in range(point_data.GetNumberOfArrays()):
                name = point_data.GetArrayName(j)
                if name != integration_time_name: # integration time is already stored
                    
                    # sort array data accordingly
                    array_data = np.array(point_data.GetArray(name), copy=True)
                    if array_data.ndim == 1:
                        array_data = array_data[idx]
                    else:
                        array_data = array_data[idx,:]
                    pl.add_attribute(t_np, array_data, name)

            self._pathlines.append(pl)
            i = i+1
            print("...finished " + str(i) + " out of " + str(n_total) + " pathlines.", end='\r')

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
        
        if velocity_gradient_name is None:
            raise AttributeError("Velocity gradient could not be computed.")
        
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
            position_attribute = pathline.get_position_attribute()
            t = position_attribute.t
            x_interp = position_attribute.interpolator

            # Determine time points at which to interpolate.
            if sampling_rate > 0:
                t0 = t[0]
                tend = t[-1]
                n_samples = int((tend-t0)/sampling_rate)+1
                ti = np.linspace(t0, tend, n_samples, endpoint=True)
            else:
                ti = t

            # Interpolate.
            xi = x_interp(ti)
            val_interp = self._flow_field.probe_field(xi, field_names)

            for field_name in field_names:
                pathline.add_attribute(ti, val_interp[field_name], 
                                       field_name, interpolation_scheme=interpolation_scheme)
            
            i = i+1
            print ("...finished " + str(i) + " out of " + str(len(self._pathlines)) + " pathlines.", end='\r')
    
    def get_flow_field(self) -> EulerianFlowField:
        """
        Returns the flow field associated with the pathline tracker.

        :return: The flow field.
        :rtype: EulerianFlowField
        """

        return self._flow_field
    
    def get_name_velocity_gradient(self) -> str | None:
        """
        Returns the name of the attribute that contains the velocity gradient. Implemented by asking the flow field.

        :return: The name of the attribute.
        :rtype: str
        """

        return self._flow_field.get_name_velocity_gradient()
    
    def get_name_omega_frame(self) -> str | None:
        """
        Returns the name of the attribute that describes the angular velocity vector of the frame of reference. Implemented by asking the flow field.

        :return: The name of the attribute.
        :rtype: str
        """

        return self._flow_field.get_name_omega_frame()
    
    def get_name_distance_center(self) -> str | None:
        """
        Returns the name of the attribute that describes the orthogonal distance to the center of rotation. Implemented by asking the flow field.

        :return: The name of the attribute.
        :rtype: str
        """

        return self._flow_field.get_name_distance_center()