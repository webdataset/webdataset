import braceexpand

from . import filters, gopen
from .handlers import reraise_exception
from .tariterators import url_opener


def cbor_iterator(data, handler=reraise_exception, info={}):
    import cbor

    for source in data:
        assert isinstance(source, dict)
        assert "stream" in source
        stream = source["stream"]
        try:
            while True:
                sample = cbor.load(stream)
                yield sample
        except EOFError:
            return None


def cbors_samples(src, handler=reraise_exception):
    streams = url_opener(src, handler=handler)
    samples = cbor_iterator(streams)
    return samples


cbors_to_samples = filters.pipelinefilter(cbors_samples)


def cbor2_iterator(data, handler=reraise_exception, info={}):
    import cbor2

    for source in data:
        assert isinstance(source, dict)
        assert "stream" in source
        stream = source["stream"]
        try:
            while True:
                sample = cbor2.load(stream)
                yield sample
        except EOFError:
            return None


def cbors2_samples(src, handler=reraise_exception):
    streams = url_opener(src, handler=handler)
    samples = cbor_iterator(streams)
    return samples


cbors2_to_samples = filters.pipelinefilter(cbors_samples)
