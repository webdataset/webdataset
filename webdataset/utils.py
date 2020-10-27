import warnings
import time

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


def add_hook(fs, f):
    assert callable(f)
    if fs is None:
        return [f]
    if isinstance(fs, list):
        return fs + [f]
    assert callable(fs)
    return [fs, f]


def call_hook(fs, *args, **kw):
    if fs is None:
        return
    if not isinstance(fs, list):
        fs = [fs]
    for f in fs:
        f(*args, **kw)


def identity(x):
    return x


def do_nothing(*args, **kw):
    """Do nothing function."""
    pass
