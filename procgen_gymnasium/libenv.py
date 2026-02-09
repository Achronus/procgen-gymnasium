"""
Vendored libenv loader — replaces gym3.libenv.CEnv.

Loads a compiled shared library (.so/.dll/.dylib) that exposes the libenv C
interface and provides numpy-backed buffers for observations, actions,
rewards, and info.
"""

import os
import platform
import ctypes
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Map libenv dtype enum values to numpy dtypes
_DTYPE_TO_NUMPY = {
    1: np.uint8,    # LIBENV_DTYPE_UINT8
    2: np.int32,    # LIBENV_DTYPE_INT32
    3: np.float32,  # LIBENV_DTYPE_FLOAT32
}

# libenv C constants (must match libenv.h)
_LIBENV_MAX_NAME_LEN = 128
_LIBENV_MAX_NDIM = 16

_SPACE_OBSERVATION = 1
_SPACE_ACTION = 2
_SPACE_INFO = 3


def get_header_dir():
    """Return path to the directory containing libenv.h."""
    return os.path.join(SCRIPT_DIR, "src")


def _load_library(lib_dir):
    """Load the shared library from lib_dir, returning a ctypes CDLL handle."""
    system = platform.system()
    if system == "Windows":
        lib_name = "env.dll"
    elif system == "Darwin":
        lib_name = "libenv.dylib"
    else:
        lib_name = "libenv.so"

    lib_path = os.path.join(lib_dir, lib_name)
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Could not find shared library at {lib_path}")

    return ctypes.CDLL(lib_path)


# ---- ctypes struct definitions matching libenv.h ----

class _LibenvValue(ctypes.Union):
    _fields_ = [
        ("uint8", ctypes.c_uint8),
        ("int32", ctypes.c_int32),
        ("float32", ctypes.c_float),
    ]


class _LibenvTensortype(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * _LIBENV_MAX_NAME_LEN),
        ("scalar_type", ctypes.c_int),
        ("dtype", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("shape", ctypes.c_int * _LIBENV_MAX_NDIM),
        ("low", _LibenvValue),
        ("high", _LibenvValue),
    ]


class _LibenvOption(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * _LIBENV_MAX_NAME_LEN),
        ("dtype", ctypes.c_int),
        ("count", ctypes.c_int),
        ("data", ctypes.c_void_p),
    ]


class _LibenvOptions(ctypes.Structure):
    _fields_ = [
        ("items", ctypes.POINTER(_LibenvOption)),
        ("count", ctypes.c_int),
    ]


class _LibenvBuffers(ctypes.Structure):
    _fields_ = [
        ("ob", ctypes.POINTER(ctypes.c_void_p)),
        ("ac", ctypes.POINTER(ctypes.c_void_p)),
        ("info", ctypes.POINTER(ctypes.c_void_p)),
        ("rew", ctypes.POINTER(ctypes.c_float)),
        ("first", ctypes.POINTER(ctypes.c_uint8)),
    ]


def _make_options(options_dict):
    """Convert a Python dict to a _LibenvOptions struct for passing to C."""
    items = []
    # keep references alive until the call completes
    keepalive = []

    for key, value in options_dict.items():
        opt = _LibenvOption()
        opt.name = key.encode("utf-8")

        if isinstance(value, bool):
            data = np.array([int(value)], dtype=np.uint8)
            opt.dtype = 1  # LIBENV_DTYPE_UINT8
            opt.count = 1
        elif isinstance(value, int):
            data = np.array([value], dtype=np.int32)
            opt.dtype = 2  # LIBENV_DTYPE_INT32
            opt.count = 1
        elif isinstance(value, float):
            data = np.array([value], dtype=np.float32)
            opt.dtype = 3  # LIBENV_DTYPE_FLOAT32
            opt.count = 1
        elif isinstance(value, str):
            data = np.frombuffer(value.encode("utf-8"), dtype=np.uint8).copy()
            opt.dtype = 1  # LIBENV_DTYPE_UINT8
            opt.count = len(data)
        else:
            raise TypeError(f"Unsupported option type {type(value)} for key '{key}'")

        opt.data = data.ctypes.data
        keepalive.append(data)
        items.append(opt)

    items_arr = (_LibenvOption * len(items))(*items)
    keepalive.append(items_arr)

    opts = _LibenvOptions()
    opts.items = items_arr
    opts.count = len(items)
    return opts, keepalive


def _get_tensortypes(lib, handle, space):
    """Query tensor types for a given space from the C library."""
    lib.libenv_get_tensortypes.restype = ctypes.c_int
    lib.libenv_get_tensortypes.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
    ]

    count = lib.libenv_get_tensortypes(handle, space, None)
    if count == 0:
        return []

    types_arr = (_LibenvTensortype * count)()
    lib.libenv_get_tensortypes(handle, space, types_arr)

    result = []
    for i in range(count):
        tt = types_arr[i]
        name = tt.name.decode("utf-8")
        np_dtype = _DTYPE_TO_NUMPY[tt.dtype]
        shape = tuple(tt.shape[j] for j in range(tt.ndim))
        result.append({
            "name": name,
            "dtype": np_dtype,
            "shape": shape,
            "scalar_type": tt.scalar_type,
            "low": tt.low,
            "high": tt.high,
            "ndim": tt.ndim,
        })
    return result


class CLibenv:
    """
    Low-level wrapper around a libenv shared library.

    Manages the C environment handle, allocates numpy buffers, and provides
    act/observe/close operations. This replaces gym3.libenv.CEnv.
    """

    def __init__(self, lib_dir, num, options, c_func_defs=None):
        self.num = num
        self._lib = _load_library(lib_dir)
        self._keepalive = []

        # Set up function signatures
        self._lib.libenv_make.restype = ctypes.c_void_p
        self._lib.libenv_make.argtypes = [ctypes.c_int, _LibenvOptions]

        self._lib.libenv_observe.restype = None
        self._lib.libenv_observe.argtypes = [ctypes.c_void_p]

        self._lib.libenv_act.restype = None
        self._lib.libenv_act.argtypes = [ctypes.c_void_p]

        self._lib.libenv_close.restype = None
        self._lib.libenv_close.argtypes = [ctypes.c_void_p]

        self._lib.libenv_set_buffers.restype = None
        self._lib.libenv_set_buffers.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_LibenvBuffers)
        ]

        # Create the environment
        opts, keepalive = _make_options(options)
        self._keepalive.extend(keepalive)
        self._handle = self._lib.libenv_make(num, opts)

        # Query tensor types
        self._ob_types = _get_tensortypes(self._lib, self._handle, _SPACE_OBSERVATION)
        self._ac_types = _get_tensortypes(self._lib, self._handle, _SPACE_ACTION)
        self._info_types = _get_tensortypes(self._lib, self._handle, _SPACE_INFO)

        # Allocate numpy buffers
        self._ob_bufs = {}
        for tt in self._ob_types:
            self._ob_bufs[tt["name"]] = np.zeros((num,) + tt["shape"], dtype=tt["dtype"])

        self._ac_bufs = {}
        for tt in self._ac_types:
            self._ac_bufs[tt["name"]] = np.zeros((num,) + tt["shape"], dtype=tt["dtype"])

        self._info_bufs = {}
        for tt in self._info_types:
            self._info_bufs[tt["name"]] = np.zeros((num,) + tt["shape"], dtype=tt["dtype"])

        self._rew_buf = np.zeros(num, dtype=np.float32)
        self._first_buf = np.zeros(num, dtype=np.uint8)

        # Set buffers on the C side
        self._set_buffers()

        # Register extra C functions (e.g. get_state, set_state)
        self._c_func_defs = c_func_defs or []

        # Observe initial state
        self._lib.libenv_observe(self._handle)

    def _set_buffers(self):
        """Pass numpy buffer pointers to the C library."""
        ob_ptrs = (ctypes.c_void_p * len(self._ob_types))()
        for i, tt in enumerate(self._ob_types):
            buf = self._ob_bufs[tt["name"]]
            for env_idx in range(self.num):
                pass  # buffers are contiguous per-env
            ob_ptrs[i] = buf.ctypes.data

        # The C side expects pointers laid out as:
        # buf[space_idx * num_envs + env_idx] = pointer to env_idx's data for space_idx
        # We need to provide per-env pointers for each space

        ob_ptr_arr = (ctypes.c_void_p * (len(self._ob_types) * self.num))()
        for space_idx, tt in enumerate(self._ob_types):
            buf = self._ob_bufs[tt["name"]]
            env_stride = int(np.prod(tt["shape"])) * buf.dtype.itemsize
            for env_idx in range(self.num):
                ob_ptr_arr[space_idx * self.num + env_idx] = buf.ctypes.data + env_idx * env_stride

        ac_ptr_arr = (ctypes.c_void_p * (len(self._ac_types) * self.num))()
        for space_idx, tt in enumerate(self._ac_types):
            buf = self._ac_bufs[tt["name"]]
            env_stride = max(int(np.prod(tt["shape"])), 1) * buf.dtype.itemsize
            for env_idx in range(self.num):
                ac_ptr_arr[space_idx * self.num + env_idx] = buf.ctypes.data + env_idx * env_stride

        info_ptr_arr = (ctypes.c_void_p * (len(self._info_types) * self.num))()
        for space_idx, tt in enumerate(self._info_types):
            buf = self._info_bufs[tt["name"]]
            env_stride = max(int(np.prod(tt["shape"])), 1) * buf.dtype.itemsize
            for env_idx in range(self.num):
                info_ptr_arr[space_idx * self.num + env_idx] = buf.ctypes.data + env_idx * env_stride

        bufs = _LibenvBuffers()
        bufs.ob = ob_ptr_arr
        bufs.ac = ac_ptr_arr
        bufs.info = info_ptr_arr
        bufs.rew = self._rew_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        bufs.first = self._first_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        # Keep references alive
        self._buf_refs = (ob_ptr_arr, ac_ptr_arr, info_ptr_arr, bufs)
        self._lib.libenv_set_buffers(self._handle, ctypes.byref(bufs))

    def act(self, action):
        """Write actions and step the environments."""
        self._ac_bufs["action"][:] = action
        self._lib.libenv_act(self._handle)

    def observe(self):
        """Read observations, rewards, and firsts from the C side."""
        self._lib.libenv_observe(self._handle)
        return (
            self._first_buf.copy(),
            {k: v.copy() for k, v in self._ob_bufs.items()},
            self._rew_buf.copy(),
            {k: v.copy() for k, v in self._info_bufs.items()},
        )

    def get_ob_bufs(self):
        """Return current observation buffers without copying."""
        return self._ob_bufs

    def get_info_bufs(self):
        """Return current info buffers without copying."""
        return self._info_bufs

    def get_reward_buf(self):
        """Return current reward buffer without copying."""
        return self._rew_buf

    def get_first_buf(self):
        """Return current first (reset) buffer without copying."""
        return self._first_buf

    def call_c_func(self, name, *args):
        """Call an extra C function registered via c_func_defs."""
        func = getattr(self._lib, name)
        return func(self._handle, *args)

    @property
    def ob_types(self):
        return self._ob_types

    @property
    def ac_types(self):
        return self._ac_types

    @property
    def info_types(self):
        return self._info_types

    def close(self):
        """Release the C environment."""
        if self._handle is not None:
            self._lib.libenv_close(self._handle)
            self._handle = None

    def __del__(self):
        self.close()
