import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time

from invoke import task

VENV = "venv"
BIN = f"{VENV}/bin"
PYTHON3 = f"{BIN}/python3"
ACTIVATE = f". {BIN}/activate;"
PIP = f"{BIN}/pip"
PACKAGE = "webdataset"
DOCKER = "wdstest"

COMMANDS = []
MODULES = [os.path.splitext(fname)[0] for fname in glob.glob(f"{PACKAGE}/*.py")]
MODULES = [re.sub("/", ".", name) for name in MODULES if name[0] != "_"]


@task
def clean(c):
    "Remove temporary files."
    c.run(f"rm -rf build site dist __pycache__ */__pycache__ *.pyc */*.pyc")


@task(clean)
def cleanall(c):
    "Remove temporary files and virtualenv."
    c.run(f"rm -rf venv")


@task
def venv(c):
    "Build the virtualenv."
    c.run(f"test -d {VENV} || python3 -m venv {VENV}")
    c.run(f"{BIN}/pip install --upgrade pip")
    c.run(f"{BIN}/pip install '.[dev]'")
    c.run(f"pre-commit install || true")
    print("done")


@task
def ruff(c):
    "Run the ruff linter."
    c.run(f"{BIN}/ruff check .")


@task
def docs(c):
    "Serve the documentation locally in a browser."
    c.run(f"{BIN}/mkdocs serve -o")


@task
def mkdocs(c):
    """Generate the documentation and push it to Github pages."""
    c.run("rm -rf site")
    c.run("mkdocs build")
    c.run("ghp-import -n -p site")
    c.run("rm -rf site")


@task
def nbgen(c):
    """Generate markdown for all example notebooks."""
    c.run("jupyter nbconvert --to markdown readme.ipynb && mv readme.md README.md")
    c.run(f"cp README.md docs/index.md")
    c.run("mkdir -p docs/examples")
    for nb in glob.glob("examples/*.ipynb"):
        output = nb.replace(".ipynb", ".md")
        if os.path.exists(output) and os.path.getmtime(nb) < os.path.getmtime(output):
            continue
        c.run(
            f"{ACTIVATE}jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {nb}"
        )
        c.run(
            f"{ACTIVATE}jupyter nbconvert {nb} --to markdown --output-dir=docs/examples"
        )


@task
def nbrun(c):
    """Run selected notebooks with papermill+parameters; put into ./out."""

    def nbprocess(c, nb, *args, **kwargs):
        """Process one notebook."""
        out_file = f"docs/output/{nb}"
        if not os.path.exists(out_file) or os.path.getmtime(nb) > os.path.getmtime(
            out_file
        ):
            c.run(
                f"../venv/bin/python -m papermill -l python {' '.join(args)} {nb} docs/output/_{nb}"
            )
            c.run(f"mv docs/output/_{nb} {out_file}")

    with c.cd("examples"):  # Change directory to 'examples'
        c.run("rm -f *.log *.out.ipynb *.stripped.ipynb _temp.ipynb", pty=True)
        c.run("mkdir -p docs/output", pty=True)
        nbprocess(c, "generate-text-dataset.ipynb")
        nbprocess(c, "train-ocr-errors-hf.ipynb", "-p", "max_steps", "100")
        nbprocess(c, "train-resnet50-wds.ipynb", "-p", "max_steps", "10000")
        nbprocess(c, "train-resnet50-wids.ipynb", "-p", "max_steps", "10000")
        nbprocess(c, "train-resnet50-multiray-wds.ipynb", "-p", "max_steps", "1000")
        nbprocess(c, "train-resnet50-multiray-wids.ipynb", "-p", "max_steps", "1000")
        nbprocess(c, "tesseract-wds.ipynb")


@task
def quick(c):
    "Run the tests."
    # venv(c)
    c.run(f"{ACTIVATE}{PYTHON3} -m pytest -x tests -m quick")


@task
def test(c):
    "Run the tests."
    # venv(c)
    c.run(f"{ACTIVATE}{PYTHON3} -m pytest -x tests")


@task
def testwids(c):
    "Run the wids tests."
    c.run(f"{ACTIVATE}{PYTHON3} -m pytest -x tests/test_wids*.py")


@task
def testdebug(c):
    "Run the tests with --pdb."
    c.run(f"{ACTIVATE}{PYTHON3} -m pytest -x --pdb tests")


@task
def testcov(c):
    "Run the tests and generate coverage.json and coverage.lcov."
    # venv(c)
    c.run(
        f"{ACTIVATE}{PYTHON3} -m pytest ./tests --cov=wids "
        + "--cov=webdataset --cov-report=term-missing --cov-branch "
        + "--cov-report=json:coverage.json --cov-report=lcov:coverage.lcov"
    )


@task
def faqmake(c):
    "Create the FAQ.md file from github issues."
    from helpers.faq import faq_intro, generate_faq_entries_from_issues, wrap_long_lines

    generate_faq_entries_from_issues()
    output = open("FAQ.md", "w")
    output.write(faq_intro)
    entries = sorted(glob.glob("faqs/[a-zA-Z]*.md"))
    entries = sorted(glob.glob("faqs/[0-9]*.md"), reverse=True)
    for fname in entries:
        with open(fname) as stream:
            text = stream.read()
        if "N/A" in text[:20]:
            continue
        text = text.strip()
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        text = wrap_long_lines(text)
        if len(text) < 10:
            continue
        text += "\n\n"
        if match := re.match(r"faqs/([0-9]+)\.md", fname):
            issue_number = int(match.group(1))
            text = f"Issue #{issue_number}\n\n{text}"
        output.write("-" * 78 + "\n\n")
        output.write(text.strip() + "\n\n")
    output.close()
    c.run("cp FAQ.md docs/FAQ.md")


# def twine_pypi_release(c):
#     "Manually push to PyPI via Twine."
#     c.run("rm -f dist/*")
#     c.run("$(PYTHON3) -m build --sdist")
#     c.run("$(PYTHON3) -m build --wheel")
#     c.run("twine check dist/*")
#     c.run("twine upload dist/*")


def update_version_numbers_locally(c):
    c.run("bump2version patch")


@task
def release(c):
    "Tag the current version as a release on Github."
    from helpers import get_changes

    assert c.run("bump2version patch").ok
    tag = "v" + open("VERSION").read().strip()
    print(f"Summarizing the changes for {tag}:")
    changes = get_changes(tag)
    print("\n---\n" + changes + "\n---\n")
    assert c.run(f"gh release create {tag} -t {tag} --notes-file -", input=changes).ok
    print(f"Release {version} created successfully.")
