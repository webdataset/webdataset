import glob
import os
import re
import shutil
import sys
import tempfile
import textwrap

from invoke import task

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
    c.run(f"{ACTIVATE}{PIP} install -r requirements.txt")
    c.run(f"{ACTIVATE}{PIP} install -r requirements.dev.txt")
    c.run(f"{ACTIVATE}{PIP} install -e .")
    print("done")


@task
def virtualenv(c):
    "Build the virtualenv."
    venv(c)


@task
def black(c):
    """Run black on the code."""
    c.run(f"{ACTIVATE}{PYTHON3} -m black webdataset wids tests examples")


@task
def autoflake(c):
    """Run autoflake on the code."""
    c.run(
        f"{ACTIVATE}{PYTHON3} -m autoflake --in-place --remove-all-unused-imports examples/[a-z]*.py webdataset/[a-z]*.py tests/[a-z]*.py wids/[a-z]*.py tasks.py"
    )

@task
def isort(c):
    """Run isort on the code."""
    c.run(f"{ACTIVATE}{PYTHON3} -m isort --atomic --float-to-top webdataset examples wids tests tasks.py")

@task
def cleanup(c):
    """Run black, autoflake, and isort on the code."""
    autoflake(c)
    isort(c)
    black(c)


@task
def pipx(c):
    """Install the package using pipx."""
    c.run(f"{ACTIVATE} pipx install -f .")
    c.run(f"{ACTIVATE} pipx inject {PACKAGE} torch")


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
    c.run(f"{ACTIVATE}{PYTHON3} -m pytest -x tests")

@task 
def debugtest(c):
    "Run the tests with --pdb."
    c.run(f"{ACTIVATE}{PYTHON3} -m pytest -x --pdb tests")

@task
def tests(c):
    "Run the tests."
    test(c)

@task
def testwids(c):
    "Run the wids tests."
    c.run(f"{ACTIVATE}{PYTHON3} -m pytest -x tests/test_wids*.py")

@task
def nbstrip(c):
    "Strip outputs from notebooks."
    for nb in glob.glob("examples/*.ipynb"):
        print("stripping", nb, file=sys.stderr)
        c.run(f"{ACTIVATE}jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {nb}")

@task
def nbexecute(c):
    print("executing notebooks, this will take a while")
    for nb in glob.glob("examples/*.ipynb"):
        print("executing", nb, file=sys.stderr)
        c.run(f"{ACTIVATE}jupyter nbconvert --execute --inplace {nb}")

@task
def newversion(c):
    """Increment the version number."""
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
    newversion(c)
    version = open("VERSION").read().strip()
    # os.system(f"hub release create {version}")  # interactive
    assert os.system("git commit -a -m 'new version'") == 0
    assert os.system("git push") == 0
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
def nbprocess(c, nb, *args, **kwargs):
    out_file = f"out/{nb}"
    if not os.path.exists(out_file) or os.path.getmtime(nb) > os.path.getmtime(out_file):
        c.run(f"../venv/bin/python -m papermill -l python {' '.join(args)} {nb} out/_{nb}")
        c.run(f"mv out/_{nb} {out_file}")

@task
def nbrun(c):
    with c.cd('examples'):  # Change directory to 'examples'
        c.run("rm -f *.log *.out.ipynb *.stripped.ipynb _temp.ipynb", pty=True)
        c.run("mkdir -p out", pty=True)

        nbprocess(c, 'generate-text-dataset.ipynb')
        nbprocess(c, 'train-ocr-errors-hf.ipynb', '-p', 'max_steps', '100')
        nbprocess(c, 'train-resnet50-wds.ipynb', '-p', 'max_steps', '10000')
        nbprocess(c, 'train-resnet50-wids.ipynb', '-p', 'max_steps', '10000')
        nbprocess(c, 'train-resnet50-multiray-wds.ipynb', '-p', 'max_steps', '1000')
        nbprocess(c, 'train-resnet50-multiray-wids.ipynb', '-p', 'max_steps', '1000')
        nbprocess(c, 'tesseract-wds.ipynb')

@task
def gendocs(c):
    "Generate docs."

    c.run("jupyter nbconvert --to markdown readme.ipynb && mv readme.md README.md")
    # convert IPython Notebooks
    # for nb in glob.glob("notebooks/*.ipynb"):
    #    c.run(f"{ACTIVATE} jupyter nbconvert {nb} --to markdown --output-dir=docsrc/.")
    #c.run(f"mkdocs build")
    #c.run(f"pdoc -t docsrc -o docs/api webdataset")
    #c.run("git add docs")


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
    "Build a base container."
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
def githubtest(c):
    "Test the latest version on Github in a docker container."
    dockerbase(c)
    docker_build(c, github_test, nocache=True)


@task
def pypitest(c):
    "Test the latest version on PyPI in a docker container."
    dockerbase(c)
    docker_build(c, pypi_test, nocache=True)


required_files = """
.github/workflows/pypi.yml
.github/workflows/test.yml
.githooks/pre-push
.gitignore
""".strip().split()

def wrap_long_lines(text, width=80, threshold=120):
    lines = text.split('\n')
    wrapped_lines = []
    for line in lines:
        if len(line) > threshold:
            wrapped_lines.append(textwrap.fill(line, width))
        else:
            wrapped_lines.append(line)
    return '\n'.join(wrapped_lines)

faq_intro = """
# WebDataset FAQ

This is a Frequently Asked Questions file for WebDataset.  It is
automatically generated from selected WebDataset issues using AI.

Since the entries are generated automatically, not all of them may
be correct.  When in doubt, check the original issue.

"""

@task
def makefaq(c):
    "Create the FAQ.md file from faqs/*.md"
    output = open("FAQ.txt", "w")
    output.write(faq_intro)
    entries = sorted(glob.glob("faqs/[a-zA-Z]*.md"))
    entries = sorted(glob.glob("faqs/[0-9]*.md"))
    for fname in entries:
        with open(fname) as stream:
            text = stream.read()
        text = text.strip()
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
        text = wrap_long_lines(text)
        if len(text) < 10:
            continue
        text += "\n\n"
        output.write("-"*78 + "\n\n")
        output.write(text.strip()+"\n\n")
    output.close()


@task
def checkall(c):
    "Check for existence of required files."
    for root, dirs, files in os.walk(f"./{PACKAGE}"):
        if "/__" in root:
            continue
        assert "__init__.py" in files, (root, dirs, files)
    assert os.path.isdir("./docs")
    for fname in required_files:
        assert os.path.exists(fname), fname
    assert "run: make" not in open(".github/workflows/test.yml").read()
