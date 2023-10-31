import argparse
import json
import os
from urllib.parse import urlparse

import braceexpand

from . import wids_dl
from . import wids


def urldir(url):
    """Return the directory part of a url."""
    parsed_url = urlparse(url)
    path = parsed_url.path
    directory = os.path.dirname(path)
    return parsed_url._replace(path=directory).geturl()


def urlfile(url):
    """Return the file part of a url."""
    parsed_url = urlparse(url)
    path = parsed_url.path
    filename = os.path.basename(path)
    return filename


def main():
    parser = argparse.ArgumentParser(
        description="Create a shard index for a set of files"
    )
    parser.add_argument("files", nargs="+", help="files to index")
    parser.add_argument("--output", "-o", help="output file name")
    parser.add_argument("--name", "-n", help="name for dataset", default="")
    parser.add_argument("--info", "-i", help="description for dataset", default=None)
    args = parser.parse_args()

    # set default output file name
    if args.output is None:
        args.output = "shardindex.json"

    # read the list of files from stdin if there is only one file and it is "-"
    if len(args.files) == 1 and files[0] == "-":
        args.files = [line.strip() for line in sys.stdin]

    # expand any brace expressions in the file names
    fnames = []
    for f in args.files:
        fnames.extend(braceexpand.braceexpand(f))

    # create the shard index
    downloader = wids_dl.SimpleDownloader()
    files = []
    for fname in fnames:
        print(fname)
        downloaded = downloader.download(fname, "/tmp/shard.tar")
        md5sum = wids.compute_file_md5sum(downloaded)
        nsamples = wids.compute_num_samples(downloaded)
        filesize = os.stat(downloaded).st_size
        files.append(
            dict(url=fname, md5sum=md5sum, nsamples=nsamples, filesize=filesize)
        )
        downloader.release(downloaded)

    # create the result dictionary
    result = dict(
        __kind__="wids-shard-index-v1",
        wids_version=1,
        shardlist=files,
    )

    if args.name != "":
        result["name"] = args.name

    # add info if it is given
    if args.info is not None:
        info = open(args.info).read()
        result["info"] = info

    # write the result
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
