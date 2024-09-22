import shutil
import tempfile

from invoke import task

base_container = f"""
FROM ubuntu:22.04
ENV LC_ALL=C
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -qqy update
RUN apt-get install -qqy git
RUN apt-get install -qqy python3
RUN apt-get install -qqy python3-pip
RUN apt-get install -qqy python3-venv
RUN apt-get install -qqy curl
WORKDIR /tmp
RUN python3 -m venv venv
RUN . venv/bin/activate; pip install --no-cache-dir pytest
# RUN . venv/bin/activate; pip install --no-cache-dir jupyterlab
# RUN . venv/bin/activate; pip install --no-cache-dir numpy
# RUN . venv/bin/activate; pip install --no-cache-dir nbconvert
RUN . venv/bin/activate; pip install --no-cache-dir torch==1.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
# RUN . venv/bin/activate; pip install --no-cache-dir torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
"""

local_test = """
FROM webdatasettest-base
ENV SHELL=/bin/bash
WORKDIR /tmp/webdataset
RUN pip install --no-cache-dir current.whl
"""


github_test = """
FROM webdatasettest-base
ENV SHELL=/bin/bash
RUN git clone https://git@github.com/tmbdev/webdataset.git /tmp/webdataset
WORKDIR /tmp/webdataset
RUN ln -s /tmp/venv .
RUN . venv/bin/activate; pip install --no-cache-dir pytest
RUN . venv/bin/activate; pip install --no-cache-dir -r requirements.txt
RUN . venv/bin/activate; python3 -m pytest
"""

pypi_test = """
FROM webdatasettest-base
ENV SHELL=/bin/bash
WORKDIR /tmp/webdataset
RUN ln -s /tmp/venv .
RUN . venv/bin/activate; pip install --no-cache-dir pytest
RUN . venv/bin/activate; pip install --no-cache-dir webdataset
RUN git clone https://git@github.com/tmbdev/webdataset.git /tmp/webdataset-github
RUN cp -av /tmp/webdataset-github/webdataset/tests tests
RUN cp -av /tmp/webdataset-github/testdata testdata
RUN . venv/bin/activate; python3 -m pytest
"""


def docker_build(c, instructions, tag=None, files=[], nocache=False):
    """Build a docker container."""
    with tempfile.TemporaryDirectory() as dir:
        with open(dir + "/Dockerfile", "w") as stream:
            stream.write(instructions)
        for fname in files:
            shutil.copy(fname, dir + "/.")
        flags = "--no-cache" if nocache else ""
        if tag is not None:
            flags += f" -t {tag}"
        c.run(f"cd {dir} && docker build {flags} .")


def here(s):
    """Return a string suitable for a shell here document."""
    return f"<<EOF\n{s}\nEOF\n"


@task
def dockerbase(c):
    """Build a base container."""
    docker_build(c, base_container, tag="webdatasettest-base")


@task
def dockerlocal(c):
    """Run tests locally in a docker container."""
    assert not "implemented"
    c.run("pip install wheel")
    c.run("rm -rf dist")
    c.run("python setup.py sdist bdist_wheel")
    c.run("cp dist/*.whl current.whl")
    c.run("cp dist/*.tar current.tar")
    docker_build(c, local_test, files=["current.whl", "current.tar"], nocache=True)


@task(dockerbase)
def dockergithubtest(c):
    "Test the latest version on Github in a docker container."
    dockerbase(c)
    docker_build(c, github_test, nocache=True)


@task
def dockerpypitest(c):
    "Test the latest version on PyPI in a docker container."
    dockerbase(c)
    docker_build(c, pypi_test, nocache=True)
