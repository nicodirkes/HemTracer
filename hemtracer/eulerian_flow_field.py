from __future__ import annotations
from vtk import vtkUnstructuredGridReader, vtkUnstructuredGrid
from vtk import vtkDoubleArray, vtkIdTypeArray, vtkPoints, vtkPolyData
from vtk import vtkThreshold, vtkProbeFilter, vtkCellDerivatives, vtkGradientFilter
from vtkmodules.util.numpy_support import numpy_to_vtk
import numpy as np
from numpy.typing import NDArray
from typing import List, Dict
from hemtracer._definitions import Vector3, vtk_point_assoc, vtk_cell_assoc

class EulerianFlowField:
    """
    A class that represents a steady flow field in Eulerian coordinates with various field data. Unsteady flow fields are currently only supported in the form of MRF data. The mesh has to be unstructured and non-moving and the flow field has to be defined as node-centered velocity field (point data in VTK). The flow field is represented by a VTK file. It has to contain a velocity field. Additional field data, e.g., an Eulerian hemolysis solution, can be imported from other VTK files using the import_field_data method. The class provides functionality to interpolate field data to pathlines.
    """

    _vtk_flow_field_name: str = None  
    """
    Name of the VTK file containing the flow field.
    """

    _vtk_flow_field_reader: vtkUnstructuredGridReader = None  
    """
    VTK reader for the flow field.
    """

    _vtk_flow_field: vtkUnstructuredGrid = None  
    """
    VTK flow field.
    """

    _velocity_name: str = "velocity"  
    """
    Name of the velocity field in the VTK file. Defaults to "velocity".
    Assume that the velocity field is called "velocity" in the VTK file.
    """

    _velocity_gradient_name: str = None  
    """
    Name of the velocity gradient field in the VTK file.
    Assume that no gradient has been computed yet.
    """

    _mrf_velocity_name: str = None  
    """
    Name of the MRF velocity field in the VTK file.
    Assume that no MRF velocity has been computed yet.
    """

    _mrf_omega_name: str = None  
    """
    Name of the angular velocity of the rotating frame of reference in the VTK file.
    Computed as part of MRF transformation.
    """

    _mrf_r_name: str = None  
    """
    Name of the orthogonal distance from the rotation axis in the VTK file.
    Computed as part of MRF transformation.
    """

    _interpolator: vtkProbeFilter = None  
    """
    Field interpolator (may be used to extract cell data along pathline).
    """

    def __init__(self, vtk_filename: str) -> None:
        """
        Initialize the EulerianField object with a VTK flow field name.

        :param vtk_filename: Name of the VTK file containing the flow field.
        :type vtk_filename: str
        """

        # Read VTK file.
        self._vtk_flow_field_name = vtk_filename
        self._vtk_flow_field_reader = vtkUnstructuredGridReader()
        self._vtk_flow_field_reader.SetFileName(self._vtk_flow_field_name)
        self._vtk_flow_field_reader.Update()
        self._vtk_flow_field = self._vtk_flow_field_reader.GetOutput()

    def set_velocity_name(self, vtk_velocity_name: str) -> None:
        """
        Specify the name of the velocity field in the VTK file (velocity transformed to absolute frame for MRF fields).
        The velocity field has to be defined as node-centered field (point data in VTK)

        :param vtk_velocity_name: The name of the velocity field in the VTK file.
        :type vtk_velocity_name: str
        """

        self._velocity_name = vtk_velocity_name

        # Check if velocity field exists.
        arr_vtk = self._vtk_flow_field.GetPointData().GetArray(self._velocity_name)
        if arr_vtk is None:
            raise ValueError("Velocity field {} not found in {}".format(self._velocity_name, self._vtk_flow_field_name))

    def set_velocity_gradient_name(self, velocity_gradient_name: str) -> None:
        """
        Specify the name of the velocity gradient field in the VTK file, if available.
        The velocity gradient field can be defined as node-centered field (point data in VTK) or cell-centered field (cell data in VTK).

        :param velocity_gradient_name: The name of the velocity gradient field in the VTK file.
        :type velocity_gradient_name: str
        """

        self._velocity_gradient_name = velocity_gradient_name

        # Check if velocity gradient field exists.
        arr_vtk_point = self._vtk_flow_field.GetPointData().GetArray(self._velocity_gradient_name)
        if arr_vtk_point is None:
            arr_vtk_cell = self._vtk_flow_field.GetCellData().GetArray(self._velocity_gradient_name)
            if arr_vtk_cell is None:
                raise ValueError("Velocity gradient field {} not found in {}".format(self._velocity_gradient_name, self._vtk_flow_field_name))

    def import_field_data(self, vtk_file_name: str, 
                          field_names_point: List[str] = None, field_names_cell: List[str] = None) -> None:
        """
        Import additional field data from a VTK file (to be interpolated to the streamlines).
        The field data has to be defined on the same mesh as the flow field.

        :param vtk_file_name: The name of the VTK file containing the additional field data.
        :type vtk_file_name: str
        :param field_names_point: A list of strings containing the names of the node-centered fields to import. If None, all fields are imported. If you do not want to import any node-centered fields, set field_names_point=[].
        :type field_names_point: List[str]
        :param field_names_cell: A list of strings containing the names of the cell-centered fields to import. If None, all fields are imported. If you do not want to import any cell-centered fields, set field_names_cell=[].
        :type field_names_cell: List[str]
        """

        vtk_reader = vtkUnstructuredGridReader()
        vtk_reader.SetFileName(vtk_file_name)
        vtk_reader.Update()
        vtk_data = vtk_reader.GetOutput()

        # Import point data.
        if field_names_point is None:
            n_arrays_point = vtk_data.GetPointData().GetNumberOfArrays()
            field_names_point = [ vtk_data.GetPointData().GetArrayName(i) for i in range(n_arrays_point) ]

        for field_name in field_names_point:

            # Check if field name already exists.
            arr_vtk = self._vtk_flow_field.GetPointData().GetArray(field_name)
            if arr_vtk is not None:
                raise ValueError("Field {} already exists in {}".format(field_name, self._vtk_flow_field_name))
            
            # Get desired field from vtk file.
            arr = vtk_data.GetPointData().GetArray(field_name)

            # Check if field exists in vtk file.
            if arr is None:
                raise ValueError("Field {} not found in {}".format(field_name, vtk_file_name))
            
            # Add field to data.
            arr_copy = vtkDoubleArray()
            arr_copy.DeepCopy(arr)
            self._vtk_flow_field.GetPointData().AddArray(arr_copy)
        
        # Import cell data.
        if field_names_cell is None:
            n_arrays_cell = vtk_data.GetCellData().GetNumberOfArrays()
            field_names_cell = [ vtk_data.GetCellData().GetArrayName(i) for i in range(n_arrays_cell) ]

        for field_name in field_names_cell:
                
            # Check if field name already exists.
            arr_vtk = self._vtk_flow_field.GetCellData().GetArray(field_name)
            if arr_vtk is not None:
                raise ValueError("Field {} already exists in {}".format(field_name, self._vtk_flow_field_name))
            
            # Get desired field from vtk file.
            arr = vtk_data.GetCellData().GetArray(field_name)

            # Check if field exists in vtk file.
            if arr is None:
                raise ValueError("Field {} not found in {}".format(field_name, vtk_file_name))
            
            # Add field to data.
            arr_copy = vtkDoubleArray()
            arr_copy.DeepCopy(arr)
            self._vtk_flow_field.GetCellData().AddArray(arr_copy)

    def compute_distance_rotation(self, omega: Vector3, x0: Vector3 = np.zeros(3)) -> None:
        """
        Compute orthogonal distance from rotation axis. This is used for the transformation of the velocity field in a rotating frame of reference and for the tank-treading morphology model with rotation correction.

        :param omega: The angular velocity vector of the rotating frame of reference.
        :type omega: Vector3
        :param x0: Some point on the rotation axis. Defaults to (0,0,0).
        :type x0: Vector3
        """

        x = np.array(self._vtk_flow_field.GetPoints().GetData(), copy=False) # point coordinates
        om_star = omega / np.linalg.norm(omega) # normalized angular velocity vector
        x0_star = x0 + np.outer(np.dot(x - x0, om_star), om_star) # projection of x onto rotation axis
        r = x - x0_star # orthogonal distance vector from rotation axis

        # Add orthogonal distance to point data.
        self._mrf_r_name = 'r_rot'
        r_array = numpy_to_vtk(r)
        r_array.SetName(self._mrf_r_name)
        self._vtk_flow_field.GetPointData().AddArray(r_array)

    def transform_mrf_data(self, rf_name: str, rf_rot: int, omega: Vector3, x0: Vector3 = np.zeros(3)) -> None:
        """
        Transform the velocity field of a rotating frame of reference.

        :param rf_name: The name of the cell-centered (!) attribute in the vtk file that specifies the reference frame ID.
        :type rf_name: str
        :param rf_rot: The reference frame ID of the rotating frame of reference.
        :type rf_rot: int
        :param omega: The angular velocity vector of the rotating frame of reference.
        :type omega: Vector3
        :param x0: Some point on the rotation axis. Defaults to (0,0,0).
        :type x0: Vector3
        """
    
        print("Transforming velocity field for MRF data...")

        omega = np.asarray(omega)
        x0 = np.asarray(x0)

        self._mrf_velocity_name = self._velocity_name + '_mrf'
        self._mrf_omega_name = 'omega_mrf'

        # Deep copy velocity in order to retrieve original velocity field later on.
        vel_data = vtkDoubleArray()
        vel_data.DeepCopy(self._vtk_flow_field.GetPointData().GetArray(self._velocity_name))
        vel_data.SetName(self._mrf_velocity_name)
        self._vtk_flow_field.GetPointData().AddArray(vel_data)

        # Write current point IDs to new array to retrieve them later on.
        vtkOrigPointIds_name = 'vtkOriginalPointIds'
        point_ids = vtkIdTypeArray()
        for i in range(self._vtk_flow_field.GetNumberOfPoints()):
            point_ids.InsertNextValue(i)
        point_ids.SetName(vtkOrigPointIds_name)
        self._vtk_flow_field.GetPointData().AddArray(point_ids)

        self.compute_distance_rotation(omega, x0)

        # Threshold to identify cells in rotating frame.
        threshold = vtkThreshold()
        threshold.SetInputData(self._vtk_flow_field)
        threshold.SetInputArrayToProcess(0, 0, 0, vtk_cell_assoc, rf_name)
        threshold.SetLowerThreshold(rf_rot-0.1)
        threshold.SetUpperThreshold(rf_rot+0.1)
        threshold.Update()
        
        r_data_rf = threshold.GetOutput().GetPointData().GetArray(self._mrf_r_name)
        pointId_data_rf = np.array(threshold.GetOutput().GetPointData().GetArray(vtkOrigPointIds_name))
        vel_data_rf = threshold.GetOutput().GetPointData().GetArray(self._velocity_name)

        # Transform velocity.
        r = np.array(r_data_rf, copy=False)
        v_rel = np.array(vel_data_rf, copy=False) - np.cross(omega,r)
        v_mrf = np.array(vel_data, copy=False)
        v_mrf[pointId_data_rf] = v_rel

        # Add angular velocity information to point data.
        omega_arr = np.zeros_like(v_mrf)
        omega_arr[pointId_data_rf] = omega
        omega_arr_vtk = numpy_to_vtk(omega_arr)
        omega_arr_vtk.SetName(self._mrf_omega_name)
        self._vtk_flow_field.GetPointData().AddArray(omega_arr_vtk)
    
    def reduce_to_relevant_fields(self, relevant_field_names: List[str]) -> None:
        """
        Specify the names of the relevant fields in the VTK file. In particular, this could be velocity and all fields you want interpolated to the pathlines, e.g., some Eulerian solution for effective shear rate or local fluid shear. All other fields are removed. This can reduce memory requirements and speed up subsequent computations.

        :param relevant_field_names: A list of strings containing the names of the relevant fields.
        :type relevant_field_names: List[str]
        """
        # Velocity is definitely relevant.
        if self._velocity_name not in relevant_field_names:
            relevant_field_names.append(self._velocity_name)

        # MRF quantities are relevant if they have been computed.
        if self._mrf_velocity_name is not None:
            if self._mrf_velocity_name not in relevant_field_names:
                relevant_field_names.append(self._mrf_velocity_name)
        
        if self._mrf_omega_name is not None:
            if self._mrf_omega_name not in relevant_field_names:
                relevant_field_names.append(self._mrf_omega_name)
        
        if self._mrf_r_name is not None:
            if self._mrf_r_name not in relevant_field_names:
                relevant_field_names.append(self._mrf_r_name)
        
        # Velocity gradient is relevant if it has been computed.
        if self._velocity_gradient_name is not None:
            if self._velocity_gradient_name not in relevant_field_names:
                relevant_field_names.append(self._velocity_gradient_name)

        # Remove all fields that are not relevant.
        pointdata = self._vtk_flow_field.GetPointData()
        for i in range(pointdata.GetNumberOfArrays()):
            name = pointdata.GetArrayName(i)
            if name not in relevant_field_names:
                pointdata.RemoveArray(name)
        celldata = self._vtk_flow_field.GetCellData()
        for i in range(celldata.GetNumberOfArrays()):
            name = celldata.GetArrayName(i)
            if name not in relevant_field_names:
                celldata.RemoveArray(name)
    
    def probe_field(self, x: List[Vector3], field_names: List[str]) -> Dict[str, NDArray]:
        """
        Extracts field information at specified points x, e.g., a pathline.

        :param x: Positions in field at which to extract information.
        :type x: List[Vector3]
        :param field_names: List of names of field in data to extract.
        :type field_names: List[str]
        :return: A dictionary where keys are field names and values are the corresponding field information.
        :rtype: Dict[str, NDArray]
        """

        # Construct data object to interpolate to.
        points = vtkPoints()
        points.SetData(numpy_to_vtk(x))
        polydata = vtkPolyData()
        polydata.SetPoints(points)

        # Construct interpolator if it has not been constructed yet.
        if self._interpolator is None:
            self._interpolator = vtkProbeFilter()
            self._interpolator.SetSourceData(self._vtk_flow_field)

        # Interpolate.
        self._interpolator.SetInputData(polydata)
        self._interpolator.Update()

        # Start dictionary with interpolated values and positions x.
        interp_values = { 'x' : x }

        # Find field either in cell values or point values and add to interp_values dict.
        for field_name in field_names:
            arr = self._interpolator.GetOutput().GetPointData().GetArray(field_name)
            if arr is None:
                arr = self._interpolator.GetOutput().GetCellData().GetArray(field_name)
                if arr is None:
                    raise ValueError("Field {} not found in {}".format(field_name, self._vtk_flow_field_name))
            interp_values[field_name] = np.array(arr , copy=True)

        return interp_values
    
    def compute_gradients(self, node_centered: bool = False) -> None:
        """
        Compute the velocity gradient tensor (if it has not been computed before). The velocity gradient tensor can be computed as cell-centered field or node-centered field. The node-centered field is more expensive to compute, but it can be interpolated to pathlines automatically. The cell-centered field is cheaper to compute, but it has to be interpolated manually to pathlines.

        :param node_centered: If True, compute the gradient as node-centered field, else compute as cell-centered. Defaults to False.
        :type node_centered: bool
        """

        if self._velocity_gradient_name is not None:
            raise RuntimeError("Velocity gradient already exists in {} or has been computed before.".format(self._vtk_flow_field_name))

        print("Computing derivatives of velocity field...")
        if node_centered:
            self._compute_gradients_point()
        else:
            self._compute_gradients_cell()

    def _compute_gradients_cell(self) -> None:
        """Compute the velocity gradient tensor as cell-centered field."""

        grad = vtkCellDerivatives()
        grad.SetInputData(self._vtk_flow_field)
        grad.SetInputArrayToProcess(0, 0, 0, vtk_point_assoc, self._velocity_name)
        grad.SetVectorModeToPassVectors()
        grad.SetTensorModeToComputeGradient()
        grad.Update()

        self._velocity_gradient_name = self._velocity_name + '_gradient'
        vtk_cell_gradient_name = 'VectorGradient'   # vtkCellDerivatives always names the output array "VectorGradient"

        # Deep copy velocity.
        grad_data = vtkDoubleArray()
        grad_data.DeepCopy(grad.GetOutput().GetCellData().GetArray(vtk_cell_gradient_name))
        grad_data.SetName(self._velocity_gradient_name)

        self._vtk_flow_field.GetCellData().AddArray(grad_data)

    def _compute_gradients_point(self) -> None:
        """Compute the velocity gradient tensor as node-centered field."""

        grad = vtkGradientFilter()
        grad.SetInputData(self._vtk_flow_field)
        grad.SetInputArrayToProcess(0, 0, 0, vtk_point_assoc, self._velocity_name)
        grad.Update()

        self._velocity_gradient_name = self._velocity_name + '_gradient'

        # Deep copy velocity.
        vtk_point_gradient_name = 'Gradients'       # vtkGradientFilter always names the output array "Gradients"
        grad_data = vtkDoubleArray()
        grad_data.DeepCopy(grad.GetOutput().GetPointData().GetArray(vtk_point_gradient_name))
        grad_data.SetName(self._velocity_gradient_name)

        self._vtk_flow_field.GetPointData().AddArray(grad_data)
    
    def get_name_advection_velocity(self) -> str:
        """
        Get the name of the velocity field relevant for advection. In case of a rotating frame of reference, this is the MRF velocity field. Otherwise, this is the velocity field.

        :return: The velocity field name.
        :rtype: str
        """
        if self._mrf_velocity_name is not None:
            return self._mrf_velocity_name
        else:
            return self._velocity_name

    def get_name_velocity_gradient(self) -> str:
        """
        Get the name of the velocity gradient field. If no velocity gradient has been computed, this is None.

        :return: The velocity gradient field name.
        :rtype: str
        """
        return self._velocity_gradient_name
    
    def get_name_omega_frame(self) -> str:
        """
        Get the name of the angular velocity of the rotating frame of reference. If no rotating frame of reference has been defined, this is None.

        :return: The angular velocity of the rotating frame of reference.
        :rtype: str
        """
        return self._mrf_omega_name
    
    def get_name_distance_center(self) -> str:
        """
        Get the name of the orthogonal distance from the rotation axis. If no rotating frame of reference has been defined, this is None.

        :return: The orthogonal distance from the rotation axis.
        :rtype: str
        """
        return self._mrf_r_name

    def get_vtk_flow_field(self) -> vtkUnstructuredGrid:
        """
        Get the VTK flow field.

        :return: The VTK flow field.
        :rtype: vtkUnstructuredGrid
        """
        return self._vtk_flow_field