import warnings
import time
import re
import importlib
import itertools as itt

__all__ = "reraise_exception ignore_and_continue ignore_and_stop warn_and_stop add_hook call_hook".split()


def reraise_exception(exn):
    """Called in an exception handler to re-raise the exception."""
    raise exn


def ignore_and_continue(exn):
    """Called in an exception handler to ignore any exception and continue."""
    return True


def warn_and_continue(exn):
    """Called in an exception handler to ignore any exception, isssue a warning, and continue."""
    warnings.warn(repr(exn))
    time.sleep(0.5)
    return True


def ignore_and_stop(exn):
    """Called in an exception handler to ignore any exception and stop further processing."""
    return False


def warn_and_stop(exn):
    """Called in an exception handler to ignore any exception and stop further processing."""
    warnings.warn(repr(exn))
    time.sleep(0.5)
    return False


def identity(x):
    """The identity function."""
    return x


def safe_eval(s, expr="{}"):
    if re.sub("[^A-Za-z0-9_]", "", s) != s:
        raise ValueError(f"safe_eval: illegal characters in: '{s}'")
    return eval(expr.format(s))


def lookup_sym(sym, modules):
    """Looks up a symbol in a list of modules."""
    for mname in modules:
        module = importlib.import_module(mname, package="webdataset")
        result = getattr(module, sym, None)
        if result is not None:
            return result
    return None


def repeatedly(loader, nepochs=999999999, nbatches=999999999999):
    """Repeatedly returns batches from a DataLoader."""
    for epoch in range(nepochs):
        for sample in itt.islice(loader, nbatches):
            yield sample
