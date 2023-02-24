from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context


def tensor_scatter_update(tensor, indices, updates, name=None):
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data

    _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "TensorScatterUpdate", name,
        tld.op_callbacks, tensor, indices, updates)
    return _result
