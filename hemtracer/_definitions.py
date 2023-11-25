from __future__ import annotations
from numpy import float64 as float
from numpy.typing import NDArray

type Vector3 = NDArray[float]   # vector with three components
type Vector9 = NDArray[float]   # vector with nine components
type Vector12 = NDArray[float]  # vector with twelve components
type Tensor3 = NDArray[float]    # 3x3 tensor

vtk_cell_assoc = "vtkDataObject::FIELD_ASSOCIATION_CELLS"
vtk_point_assoc = "vtkDataObject::FIELD_ASSOCIATION_POINTS"