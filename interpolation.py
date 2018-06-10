# -*- coding: utf-8 -*-
import sys
from os.path import abspath, dirname, join as pjoin
import logging
import warnings
from functools import partial

from enum import IntEnum
from ctypes import (c_int, c_double, c_void_p, Structure, byref)
import numpy as np
import xarray as xr

from typing import Iterable

logging.basicConfig(format='%(levelname)s:%(message)s',
                    level=logging.ERROR)
logger = logging.getLogger(__name__)

try:
    from sdf import ndtable
except ModuleNotFoundError:
    logger.error("*" * 80)
    logger.error("\tModule sdf not found, install with:")
    logger.error("\t\tpip install --no-deps sdf")
    logger.error("*" * 80)
    raise

from xarray.core.extensions import AccessorRegistrationWarning

class LookUpEnum(IntEnum):
    """LookUpEnum provide from_param method for ctypes.
    """
    @classmethod
    def from_param(cls, obj):
        return int(obj)

ARRAY_MAX_NDIMS = c_int * np.MAXDIMS
POINTER_TO_DOUBLE = np.ctypeslib.ndpointer(dtype='f8',
                                           flags='C_CONTIGUOUS')
POINTER_TO_BP = POINTER_TO_DOUBLE * np.MAXDIMS

class NDTable_h(Structure):
    """
    Parameter : xarray instance

    https://git.io/vh3af
    MAX_NDIMS == np.MAXDIMS == 32
    """

    _fields_ = [("ndim", c_int),                  # ndims
                ("shape", c_int * np.MAXDIMS),    # dims[MAX_NDIMS]
                ("size", c_int),                  # numel
                ("strides", c_int * np.MAXDIMS),  # offs[MAX_NDIMS]
                ("data", POINTER_TO_DOUBLE),      # *data
                ("breakpoints", POINTER_TO_DOUBLE * np.MAXDIMS)]   # *scales[MAX_NDIMS]

    INTERP_METH = LookUpEnum('INTERP_METH',
                             'hold nearest linear akima fritsch_butland steffen')
    EXTRAP_METH = LookUpEnum('EXTRAP_METH',
                             'hold linear')
    def __init__(self, obj: xr.DataArray) -> None:
        assert isinstance(obj, xr.DataArray), \
            f"{obj.__class__.__name__} not supported"

        data = obj.data.astype('f8')
        kwargs = {
            'ndim': obj.ndim,
            'shape': self._fields_[1][1](*obj.shape),
            'size': data.size,
            'strides': ARRAY_MAX_NDIMS(*data.strides),
            'data': data.ctypes.data_as(POINTER_TO_DOUBLE),
            'breakpoints': POINTER_TO_BP(\
                *[np.asanyarray(obj.coords[elt], dtype='f8',
                                order='C').ctypes.data for elt in obj.dims])
        }
        super(NDTable_h, self).__init__(**kwargs)

    @classmethod
    def from_param(cls, obj):
        return byref(obj)


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=AccessorRegistrationWarning)

    @xr.register_dataarray_accessor('__call__')
    @xr.register_dataarray_accessor('interpolate')
    class _interpolate(object):
        def __init__(self, obj):
            """Interpolation

            Parameters
            ----------
            """
            self._obj = obj

            # Import evaluate_interpolation
            _arch = '64' if sys.maxsize > 2 ** 32 else '32'
            lib_path = dirname(ndtable.__file__)

            if sys.platform.startswith('darwin'):
                lib_path = pjoin(lib_path, 'darwin64')
                lib_name = 'libNDTable.dylib'
            elif sys.platform.startswith('win'):
                lib_path = pjoin(lib_path, 'win' + _arch)
                lib_name = 'ndtable.dll'
            elif sys.platform.startswith('linux'):
                lib_path = pjoin(lib_path, 'linux' + _arch)
                lib_name = 'libndtable.so'
            else:
                raise NotImplementedError(f"Unsupported platform: {sys.platform}")

            self._libndtable = np.ctypeslib.load_library(lib_name, lib_path)
            self._libndtable.evaluate.argtypes = [NDTable_h, c_int, c_void_p,
                                                 c_void_p, c_int, c_int, c_void_p]
            self._libndtable.evaluate.restype = c_int

            for m in NDTable_h.INTERP_METH.__members__.keys():
                setattr(self, m, partial(self.__call__, interp=m))

        def __call__(self, *points, interp='linear', extrap='hold', **kwargs):
            assert len(set(self._obj.dims) & set(kwargs)) + len(points) == self._obj.ndim, \
                "Not enough dimensions for interpolation"

            # Convert to tuple
            points = list(points)

            # mix usage points/kwargs
            args = {dim: kwargs[dim] if dim in kwargs else points.pop(0)
                    for dim in self._obj.dims}

            # Compute args dimensions and check compatibility without
            # broadcasting rules.
            dims = np.array([len(args[_k]) if "__len__" in dir(args[_k])
                             else 1 for _k in args])
            assert all((dims == max(dims)) + (dims == 1)), "problème"

            #  - create a clean argument list
            args = [np.asarray(args[_x], 'f8')
                    if "__len__" in dir(args[_x])
                    else np.ones((max(dims),), 'f8') * args[_x]
                    for _x in self._obj.dims]

            values = np.empty(args[0].shape)

            c_params_p = c_void_p * np.MAXDIMS  # len(self._obj.dims)

            res = self._libndtable.evaluate(NDTable_h(self._obj),
                                            c_int(self._obj.ndim),
                                            c_params_p(*[_a.ctypes.get_as_parameter()
                                                         for _a in args]),
                                            NDTable_h.INTERP_METH[interp],
                                            NDTable_h.EXTRAP_METH[extrap],
                                            c_int(values.size),
                                            values.ctypes.get_as_parameter())

            assert res == 0, 'An error occurred during interpolation'

            return values[0] if len(values) == 1 else values
