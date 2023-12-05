import os
import json
import io
from urllib.parse import urlparse, urljoin, urlunparse
import tempfile
from .wids_dl import SimpleDownloader


def load_remote_spec(source):
    """Load a remote or local dataset description in JSON format,
    using the Python web client APIs."""

    if isinstance(source, str):
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = SimpleDownloader()
            dlname = os.path.join(tmpdir, "dataset.json")
            local = downloader.download(source, dlname)
            with open(local) as f:
                dsdesc = json.load(f)
            downloader.release(local)
    elif isinstance(source, io.IOBase):
        dsdesc = json.load(source)
    else:
        # FIXME: use gopen
        import requests

        jsondata = requests.get(source).text
        dsdesc = json.loads(jsondata)
    return dsdesc


def urldir(url):
    """Return the directory part of a url."""
    parsed_url = urlparse(url)
    path = parsed_url.path
    directory = os.path.dirname(path)
    return parsed_url._replace(path=directory).geturl()


def urlmerge(base, url):
    """
    Merges a base URL and a relative URL.

    The function fills in any missing part of the url from the base,
    except for params, query, and fragment, which are taken only from the 'url'.
    For the pathname component, it merges the paths like os.path.join:
    an absolute path in 'url' overrides the base path, otherwise the paths are merged.

    Parameters:
    base (str): The base URL.
    url (str): The URL to merge with the base.

    Returns:
    str: The merged URL.
    """
    # Parse the base and the relative URL
    parsed_base = urlparse(base)
    parsed_url = urlparse(url)

    # Merge paths using os.path.join
    # If the url path is absolute, it overrides the base path
    if parsed_url.path.startswith("/"):
        merged_path = parsed_url.path
    else:
        merged_path = os.path.normpath(os.path.join(parsed_base.path, parsed_url.path))

    # Construct the merged URL
    merged_url = urlunparse(
        (
            parsed_url.scheme or parsed_base.scheme,
            parsed_url.netloc or parsed_base.netloc,
            merged_path,
            parsed_url.params,  # Use params from the url only
            parsed_url.query,  # Use query from the url only
            parsed_url.fragment,  # Use fragment from the url only
        )
    )

    return merged_url


def load_remote_shardlist(source, *, options={}, base=None):
    spec = load_remote_spec(source)
    spec = dict(spec, **options)
    shardlist = extract_shardlist(spec)
    if base is None and isinstance(source, str):
        base = urldir(source)
    elif base is True:
        base = spec.get("base")
    if isinstance(base, str):
        for shard in shardlist:
            shard["url"] = urlmerge(base, shard["url"])
    verbose = int(os.environ.get("WIDS_VERBOSE", "0"))
    if verbose >= 1:
        print("WIDS base", base)
        if verbose >= 2:
            print("WIDS shards", shardlist)
    return shardlist


def check_shards(l):
    """Check that a list of shards is well-formed.

    This checks that the list is a list of dictionaries, and that
    each dictionary has a "url" and a "nsamples" key.
    """
    assert isinstance(l, list)
    for shard in l:
        assert isinstance(shard, dict)
        assert "url" in shard
        assert "nsamples" in shard
    return l


def set_all(l, k, v):
    """Set a key to a value in a list of dictionaries."""
    if v is None:
        return
    for x in l:
        if k not in x:
            x[k] = v


def extract_shardlist(dsdesc):
    """Extract a list of shards from a dataset description.
    Dataset descriptions are JSON files. They must have the following format;

    {
        "wids_version": 1,
        # optional immediate shardlist
        "shardlist": [
            {"url": "http://example.com/file.tar", "nsamples": 1000},
            ...
        ],
        # sub-datasets
        "datasets": [
            {"source_url": "http://example.com/dataset.json"},
            {"shardlist": [
                {"url": "http://example.com/file.tar", "nsamples": 1000},
                ...
            ]}
            ...
        ]
    }
    """
    assert isinstance(dsdesc, dict)
    shardlist = dsdesc.get("shardlist", [])
    set_all(shardlist, "weight", dsdesc.get("weight"))
    check_shards(shardlist)
    assert "wids_version" in dsdesc, "No wids_version in dataset description"
    assert dsdesc["wids_version"] == 1, "Unknown wids_version"
    for component in dsdesc.get("datasets", []):
        for i in range(10):
            if "source_url" not in component:
                break
            component = load_remote(component["source_url"])
        assert i < 9, "Too many levels of indirection"
        if "shardlist" in component:
            l = check_shards(component["shardlist"])
            set_all(l, "weight", component.get("weight"))
            shardlist.extend(l)
    assert len(shardlist) > 0, "No shards found"
    return shardlist
