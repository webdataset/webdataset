#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

"""Pluggable exception handlers.

These are functions that take an exception as an argument and then return...

- the exception (in order to re-raise it)
- True (in order to continue and ignore the exception)
- False (in order to ignore the exception and stop processing)

They are used as handler= arguments in much of the library.
"""

import time
import warnings


def reraise_exception(exn):
    """Re-raise the given exception.

    Args:
        exn: The exception to be re-raised.

    Raises:
        The input exception.
    """
    raise exn


def ignore_and_continue(exn):
    """Ignore the exception and continue processing.

    Args:
        exn: The exception to be ignored.

    Returns:
        bool: Always returns True to indicate continuation.
    """
    return True


def warn_and_continue(exn):
    """Issue a warning for the exception and continue processing.

    Args:
        exn: The exception to be warned about.

    Returns:
        bool: Always returns True to indicate continuation.
    """
    warnings.warn(repr(exn))
    time.sleep(0.5)
    return True


def ignore_and_stop(exn):
    """Ignore the exception and stop further processing.

    Args:
        exn: The exception to be ignored.

    Returns:
        bool: Always returns False to indicate stopping.
    """
    return False


def warn_and_stop(exn):
    """Issue a warning for the exception and stop further processing.

    Args:
        exn: The exception to be warned about.

    Returns:
        bool: Always returns False to indicate stopping.
    """
    warnings.warn(repr(exn))
    time.sleep(0.5)
    return False
