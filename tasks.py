from invoke import task
import os
import re
import sys
import tempfile
import shutil
import glob

ACTIVATE = ". ./venv/bin/activate;"
PACKAGE = "webdataset"
VENV = "venv"
PYTHON3 = f"{VENV}/bin/python3"
PIP = f"{VENV}/bin/pip"
TEMP = "webdataset.yml"
DOCKER = "wdstest"

COMMANDS = []
MODULES = [os.path.splitext(fname)[0] for fname in glob.glob(f"{PACKAGE}/*.py")]
MODULES = [re.sub("/", ".", name) for name in MODULES if name[0] != "_"]


@task
def venv(c):
    "Build the virtualenv."
    c.run("git config core.hooksPath .githooks")
    c.run(f"test -d {VENV} || python3 -m venv {VENV}")
    # c.run(f"{ACTIVATE}{PIP} install torch torchvision")
    # c.run(f"{ACTIVATE}{PIP} install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html")
    # c.run(f"{ACTIVATE}{PIP} install torch==1.8.2+cu102 torchvision==0.9.2+cu102 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html")
    c.run(f"{ACTIVATE}{PIP} install -r requirements.dev.txt")
    c.run(f"{ACTIVATE}{PIP} install -r requirements.txt")
    print("done")


@task
def virtualenv(c):
    "Build the virtualenv."
    venv(c)
    
@task
def black(c):
    c.run(f"{ACTIVATE}{PYTHON3} -m black webdataset")
    
@task
def autoflake(c):
    c.run(f"{ACTIVATE}{PYTHON3} -m autoflake --in-place --remove-all-unused-imports webdataset/[a-z]*.py webdataset/tests/[a-z]*.py")


@task
def minenv(c):
    "Build the virtualenv (minimal)."
    c.run("git config core.hooksPath .githooks")
    c.run(f"test -d {VENV} || python3 -m venv {VENV}")
    c.run(f"{ACTIVATE}{PIP} install -r requirements.txt")
    print("done")


@task
def test(c):
    "Run the tests."
    # venv(c)
    c.run(f"{ACTIVATE}{PYTHON3} -m pytest -x")


@task
def newversion(c):
    """Increment the version number."""
    if "working tree clean" not in c.run("git status").stdout:
        input()
    text = open("setup.py").read()
    version = re.search('version *= *"([0-9.]+)"', text).group(1)
    print("old version", version)
    version = [int(x) for x in version.split(".")]
    version[-1] += 1
    version = ".".join(str(x) for x in version)
    print("new version", version)
    text = re.sub(
        r'version *= *"[0-9]+[.][0-9]+[.][0-9]+"',
        f'version = "{version}"',
        text,
    )
    with open("setup.py", "w") as stream:
        stream.write(text)
    with open("VERSION", "w") as stream:
        stream.write(version)
    text = open("webdataset/__init__.py").read()
    text = re.sub(
        r'^__version__ = ".*',
        f'__version__ = "{version}"',
        text,
        flags=re.MULTILINE,
    )
    with open("webdataset/__init__.py", "w") as stream:
        stream.write(text)
    os.system("grep 'version *=' setup.py")
    os.system("grep '__version__ *=' webdataset/__init__.py")
    # venv(c)
    # c.run(f"{ACTIVATE}{PYTHON3} -m pytest")
    # c.run("git add VERSION setup.py webdataset/__init__.py")
    # c.run("git commit -m 'incremented version'")
    # c.run("git push")


@task
def release(c):
    "Tag the current version as a release on Github."
    if "working tree clean" not in c.run("git status").stdout:
        input()
    version = open("VERSION").read().strip()
    # os.system(f"hub release create {version}")  # interactive
    os.system(f"gh release create {version}")  # interactive


@task
def coverage(c):
    """Run tests and test coverage."""
    c.run("coverage run -m pytest && coveragepy-lcov")


pydoc_template = """
# Module `{module}`

```
{text}
```
"""

command_template = """
# Command `{command}`

```
{text}
```
"""


@task
def nbgen(c):
    "Reexecute IPython Notebooks."
    opts = "--ExecutePreprocessor.timeout=-1"
    for nb in glob.glob("notebooks/*.ipynb"):
        if "/convert-" in nb:
            continue
        c.run(f"{ACTIVATE} jupyter nbconvert {opts} --execute --to notebook {nb}")


@task
def gendocs(c):
    "Generate docs."

    c.run("jupyter nbconvert --to markdown readme.ipynb && mv readme.md README.md")
    # convert IPython Notebooks
    for nb in glob.glob("notebooks/*.ipynb"):
        c.run(f"{ACTIVATE} jupyter nbconvert {nb} --to markdown --output-dir=docsrc/.")
    c.run(f"mkdocs build")
    c.run(f"pdoc -t docsrc -o docs/api webdataset")
    c.run("git add docs")


@task
def clean(c):
    "Remove temporary files."
    c.run(f"rm -rf {TEMP}")
    c.run(f"rm -rf build dist __pycache__ */__pycache__ *.pyc */*.pyc")


@task(clean)
def cleanall(c):
    "Remove temporary files and virtualenv."
    c.run(f"rm -rf venv")


@task(test)
def twine_pypi_release(c):
    "Manually push to PyPI via Twine."
    c.run("rm -f dist/*")
    c.run("$(PYTHON3) setup.py sdist bdist_wheel")
    c.run("twine check dist/*")
    c.run("twine upload dist/*")


base_container = f"""
FROM ubuntu:20.04
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
RUN . venv/bin/activate; pip install --no-cache-dir jupyterlab
RUN . venv/bin/activate; pip install --no-cache-dir numpy
RUN . venv/bin/activate; pip install --no-cache-dir nbconvert
RUN . venv/bin/activate; pip install --no-cache-dir torch==1.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN . venv/bin/activate; pip install --no-cache-dir torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
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
    return f"<<EOF\n{s}\nEOF\n"


@task
def dockerbase(c):
    "Build a base container."
    docker_build(c, base_container, tag="webdatasettest-base")


@task(dockerbase)
def githubtest(c):
    "Test the latest version on Github in a docker container."
    dockerbase(c)
    docker_build(c, github_test, nocache=True)


@task
def pypitest(c):
    "Test the latest version on PyPI in a docker container."
    dockerbase(c)
    docker_build(c, pypi_test, nocache=True)


required_files = f"""
.github/workflows/pypi.yml
.github/workflows/test.yml
.githooks/pre-push
.gitignore
""".strip().split()


@task
def checkall(c):
    "Check for existence of required files."
    for (root, dirs, files) in os.walk(f"./{PACKAGE}"):
        if "/__" in root:
            continue
        assert "__init__.py" in files, (root, dirs, files)
    assert os.path.isdir("./docs")
    for fname in required_files:
        assert os.path.exists(fname), fname
    assert "run: make" not in open(".github/workflows/test.yml").read()
