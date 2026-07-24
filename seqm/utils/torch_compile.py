import warnings

import torch

_COMPILE_CONTROL_KEYS = {"enabled", "options"}
_COMPILE_SELECTION_KEYS = {"target", "compile_omx", "compile_cis", "compile_fock", "compile_nac"}


def _compile_config(enabled):
    return {"enabled": enabled, "options": {}}


def normalize_torch_compile_config(seqm_parameters, override=None):
    """
    Normalize the torch.compile configuration used by dynamics drivers.

    Accepted forms:
      torch_compile=False
      torch_compile=True
      torch_compile={"enabled": True, "mode": "reduce-overhead"}
      torch_compile={"enabled": True, "options": {"mode": "reduce-overhead"}}
    """
    raw = seqm_parameters.get("torch_compile", False) if override is None else override
    if raw is None or raw is False:
        return _compile_config(False)
    if raw is True:
        return _compile_config(True)
    if not isinstance(raw, dict):
        raise TypeError("torch_compile must be a bool or a configuration dict.")
    if _COMPILE_SELECTION_KEYS.intersection(raw):
        raise ValueError(
            "torch_compile does not accept kernel selection; PYSEQM selects kernels automatically."
        )

    enabled = bool(raw.get("enabled", True))
    options = dict(raw.get("options", {}))
    for key, value in raw.items():
        if key not in _COMPILE_CONTROL_KEYS:
            options[key] = value
    return {"enabled": enabled, "options": options}


class OptionalCompiledFunction:
    """torch.compile wrapper for function-level kernels with eager fallback."""

    def __init__(self, fn, compile_options=None, label=None):
        self.fn = fn
        self.compile_options = {} if compile_options is None else dict(compile_options)
        self.label = label or getattr(fn, "__name__", "compiled_function")
        self._compiled_fn = None
        self._compile_disabled = False
        self._warned = False

    def _compiled(self):
        if self._compile_disabled:
            return None
        if self._compiled_fn is not None:
            return self._compiled_fn
        if not hasattr(torch, "compile"):
            self._compile_disabled = True
            warnings.warn(
                "torch.compile was requested but this PyTorch build does not provide it; "
                f"running {self.label} without compilation.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        self._compiled_fn = torch.compile(self.fn, **self.compile_options)
        return self._compiled_fn

    def __call__(self, *args, **kwargs):
        try:
            compiled = self._compiled()
            if compiled is None:
                return self.fn(*args, **kwargs)
            return compiled(*args, **kwargs)
        except Exception as exc:
            self._compile_disabled = True
            if not self._warned:
                warnings.warn(
                    f"torch.compile failed for {self.label}; falling back to eager execution. "
                    f"Original error: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned = True
            return self.fn(*args, **kwargs)


def optional_compile_function(fn, compile_options=None, label=None):
    return OptionalCompiledFunction(fn, compile_options=compile_options, label=label)
