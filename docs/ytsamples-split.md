# Getting Started

This worksheet shows how you can preprocess and split videos for deep learning. In these examples, we use Google Cloud Storage. In order to actually run this worksheet, you need:

- [Install Google's `gsutil` program](https://cloud.google.com/storage/docs/gsutil_install)
- [Install the `tarp` utility](https://github.com/tmbdev/tarp)

But it's not necessary to run the worksheet; you can follow it and apply the principles to whatever cloud, object, or local storage you use.

# Sharded YouTube Files

We have about 6M videos downloaded from YouTube (part of the YouTube 8m dataset released by Google).

These videos are stored as 3000 shards, each containing about 2000 videos and each about 56 Gbytes large.

There is just one sample shard stored in the public bucket `gs://nvdata-ytsamples`.


```python
!gsutil ls gs://nvdata-ytsamples/yt8m-lo-*.tar | head
```

    gs://nvdata-ytsamples/yt8m-lo-000000.tar


Each shard contains the video itself (`.mp4`) plus a lot of associated metadata.


```python
!gsutil cat gs://nvdata-ytsamples/yt8m-lo-000000.tar | tar tf - | head
```

    ---2pGwkL7M.annotations.xml
    ---2pGwkL7M.description
    ---2pGwkL7M.dllog
    ---2pGwkL7M.info.json
    ---2pGwkL7M.mp4
    ---2pGwkL7M.txt
    -2cScG5TqjQ.annotations.xml
    -2cScG5TqjQ.description
    -2cScG5TqjQ.dllog
    -2cScG5TqjQ.info.json


# Splitting the Videos

For training, we usually don't want to use the entire YouTube video at once; they are variable length and are hard to fit into the GPU. Instead, we want to split up the video into frames or short clips.

Here, we are using a script based on ffmpeg to split up each video into a set of clips. We also rescale all the images to a common size. The input video for this script is assumed to be `sample.mp4`, and the output clips will be stored in files like `sample-000000.mp4`.


```python
%%writefile extract-segments.sh
#!/bin/bash
exec > $(dirname $0)/_extract-segments-last.log 1>&2
#ps auxw --sort -vsz | sed 5q

set -e
set -x
set -a

size=${size:-256:128}
duration=${duration:-2}
count=${count:-999999999} 

# get mp4 metadata (total length, etc.)
ffprobe sample.mp4 -v quiet -print_format json -show_format -show_streams > sample.mp4.json

# perform the rescaling and splitting
ffmpeg -loglevel error -stats -i sample.mp4 \
    -vf "scale=$size:force_original_aspect_ratio=decrease,pad=$size:(ow-iw)/2:(oh-ih)/2" \
    -c:a copy -f segment -segment_time $duration -reset_timestamps 1  \
    -segment_format_options movflags=+faststart \
    sample-%06d.mp4

# copy the metadata into each video fragment
for s in sample-??????.mp4; do
    b=$(basename $s .mp4)
    cp sample.mp4.json $b.mp4.json || true
    cp sample.info.json $b.info.json || true
done
```

    Writing extract-segments.sh



```python
!chmod 755 ./extract-segments.sh
```

# Running the Script over All Videos in a Shard

Next, we use the `tarp` command to iterate the above script over each `.mp4` file in shard `000000`.


```python
!tarp proc --help
```

    Usage:
      tarp [OPTIONS] proc [proc-OPTIONS] [Inputs...]
    
    Application Options:
      -v                      verbose output
    
    Help Options:
      -h, --help              Show this help message
    
    [proc command options]
          -f, --field=        fields to extract; name or name=old1,old2,old3
          -o, --outputs=      output file
              --slice=        slice of input stream
          -c, --command=      shell command running in each sample dir
          -m, --multicommand= shell command running in each sample dir
              --shell=        shell command running in each sample dir (default:
                              /bin/bash)
    



```python
%%writefile splitit.sh
gsutil cat gs://lpr-yt8m-lo-sharded/yt8m-lo-000000.tar |
tarp proc -m $(pwd)/extract-segments.sh - -o - |
tarp split - -o yt8m-clips-%06d.tar
```

    Writing splitit.sh


We can now run this script using:

```Bash
$ bash ./splitit.sh
```

It's best to do this outside Jupyter since Jupyter doesn't work with long running shell jobs.

Of course, if you want to run this code over all 3000 shards, you probably will want to submit jobs based on `splitit.sh` to some job queuing system.

Also note that the `--post` option to `tarp split` lets us upload output shards as soon as they have been created, allowing the script to work with very little local storage.

# Output

Let's have a look at the output to make sure it's OK.


```python
!tar tf yt8m-clips-000000.tar | head
```

    ---2pGwkL7M/000000.mp4.json
    ---2pGwkL7M/000000.info.json
    ---2pGwkL7M/000000.mp4
    ---2pGwkL7M/000001.info.json
    ---2pGwkL7M/000001.mp4
    ---2pGwkL7M/000001.mp4.json
    ---2pGwkL7M/000002.info.json
    ---2pGwkL7M/000002.mp4
    ---2pGwkL7M/000002.mp4.json
    ---2pGwkL7M/000003.info.json
    tar: write error


These clips have been uploaded to `gs://nvdata-ytsamples/yt8m-clips-{000000..000009}.tar`.

# Processing Many Shards in Parallel

One of the benefits of sharding is that you can process them in parallel. The YT8m-lo dataset consists of 3000 shards with about 2000 videos each. To process them in parallel, you can use standard tools. You need to modify the script a little to take a shard name and to upload the result into a common bucket.

```Bash
# splitshard.sh
gsutil cat gs://lpr-yt8m-lo-sharded/yt8m-lo-$1.tar |
tarp proc -m $(pwd)/extract-segments.sh - -o - |
tarp split - -o yt8m-clips-$1-%06d.tar -p 'gsutil cp %s gs://my-output/%s && rm -f %s`
```

On a single machine, you could run run this script with:

```Bash
for shard in {000000..003000}; do echo $shard; done |
xargs -i bash splitshard.sh $shard
```

More likely, you are going to run a job of this size in the cluster, using Kubernetes or some other job queuing program. Generally, you need to create a Docker container containing the above script (and all dependencies), and then submit it with a simple loop again:

```Bash
for shard in {000000..003000}; do echo $shard; done |
submit-job --container=splitshard-container $1
```

If you store your shards on AIStore, you can also simply tell AIStore to transform a set of shards using the `splitshard-container` for you.


```python

```
