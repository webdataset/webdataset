#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

"""A small curry wrapper for the functions in the `iterators` package."""


from . import iterators


class Curried2(object):
    """Helper class for currying pipeline stages.

    We use this roundabout construct becauce it can be pickled.
    """

    def __init__(self, f, *args, **kw):
        """Create a curried function."""
        self.f = f
        self.args = args
        self.kw = kw

    def __call__(self, data):
        """Call the curried function with the given argument."""
        return self.f(data, *self.args, **self.kw)

    def __str__(self):
        """Compute a string representation."""
        return f"<{self.f.__name__} {self.args} {self.kw}>"

    def __repr__(self):
        """Compute a string representation."""
        return f"<{self.f.__name__} {self.args} {self.kw}>"


class Curried(object):
    """Helper class for currying pipeline stages.

    We use this roundabout construct because it can be pickled.
    """

    def __init__(self, f):
        """Store the function for future currying."""
        self.f = f

    def __call__(self, *args, **kw):
        """Curry with the given arguments."""
        return Curried2(self.f, *args, **kw)


info = Curried(iterators.info)
shuffle = Curried(iterators.shuffle)
select = Curried(iterators.select)
decode = Curried(iterators.decode)
map = Curried(iterators.map)
rename = Curried(iterators.rename)
associate = Curried(iterators.associate)
map_dict = Curried(iterators.map_dict)
to_tuple = Curried(iterators.to_tuple)
map_tuple = Curried(iterators.map_tuple)
batched = Curried(iterators.batched)
unbatched = Curried(iterators.unbatched)
