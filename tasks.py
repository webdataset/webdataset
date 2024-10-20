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

import yaml
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
    c.run(f"{BIN}/pip install -e '.[dev]'")
    c.run(f"pre-commit install || true")
    print("done")


@task
def ruff(c):
    "Run the ruff linter."
    c.run(f"{BIN}/ruff check .")


@task
def docsserve(c):
    "Serve the documentation locally in a browser."
    c.run(f"{BIN}/mkdocs serve -o")


@task
def docspush(c):
    """Generate the documentation and push it to Github pages."""
    c.run("rm -rf site")
    c.run("mkdocs build")
    c.run("ghp-import -n -p site")
    c.run("rm -rf site")


def summarize_notebook(nb):
    """Summarize a notebook."""
    import textwrap

    prompt = textwrap.dedent(
        """
    Here is a notebook in markdown format. Please summarize the purpose and contents
    of the notebook in a few sentences. The only markup you may use is `...` for
    quoting identifiers. Except for quoted identifiers, do not include any code 
    or output in the summary. Do not use any other markup or markdown, just plain text.
    In your summary, focus on the use of webdataset, wids, or wsds libraries (note:
    these are different libraries and be sure to talk only about the library that
    is being used in the notebook) in the notebooks and what
    the notebook illustrates about the use of those libraries,
    rather than the deep learning problem or processing problem used to illustrate
    the library usage. Mention the primary classes in those libraries used/exemplified
    by each notebook.
    Keep your summary brief, 1-3 sentences at most. Do not describe the contents
    of the notebook step-by-step.
    """
    )
    summary = os.popen(f"sgpt --no-md '{prompt}' < {nb}").read().strip()
    summary = textwrap.fill(summary, 80)
    return summary


def find_with_key(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for k, v in d.items():
            result = find_with_key(v, key)
            if result is not None:
                return result
    if isinstance(d, list):
        for v in d:
            result = find_with_key(v, key)
            if result is not None:
                return result
    return None


@task
def docsnbgen(c):
    assert os.path.exists("./mkdocs.yml")
    assert os.path.exists("./docs")
    assert os.path.exists("./examples")
    structure = yaml.safe_load(open("mkdocs.yml"))
    structure = find_with_key(structure, "Examples")
    for item in structure:
        if not isinstance(item, dict):
            continue
        k = list(item.keys())[0]
        v = item[k]
        odir = f"./docs/examples/{k}"
        os.makedirs(odir, exist_ok=True)
        for onav in v:
            if "index.md" in onav:
                continue
            output = f"./docs/{onav}"
            nb = "./examples/" + os.path.basename(output).replace(".md", ".ipynb")
            print(nb, "-->", output)
            # continue
            if os.path.exists(output) and os.path.getmtime(nb) < os.path.getmtime(
                output
            ):
                continue
            c.run(
                f"{ACTIVATE}jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {nb}"
            )
            c.run(
                f"{ACTIVATE}jupyter nbconvert {nb} --to markdown --output-dir=docs/examples/{k}"
            )
            summary = summarize_notebook(output)
            summary_fname = output.replace(".md", ".summary.md")
            with open(summary_fname, "w") as stream:
                stream.write(summary)
            print()

        def mksection(summary_fname):
            with open(summary_fname) as stream:
                summary = stream.read().strip()
            section_name = os.path.basename(summary_fname).replace(".summary.md", "")
            capitalized_name = section_name.replace("-", " ").title()
            link = f"[{capitalized_name}](./{section_name})"
            return f"### {capitalized_name}\n\n{link}\n\n{summary}\n\n"

        summaries = [
            mksection(fname) for fname in glob.glob(f"docs/examples/{k}/*.summary.md")
        ]
        with open(f"docs/examples/{k}/index.md", "w") as stream:
            print("Writing", f"docs/examples/{k}/index.md")
            stream.write("\n\n".join(summaries))


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
def releasenotes(c):
    # get the last release tag using gh
    last_tag = c.run("gh release list --limit 1 | cut -f1").stdout.strip()
    print("Last tag:", last_tag)
    # compute a diff between the last tag and the current state
    diff = c.run(f"gh release view {last_tag}").stdout
    cmd = "git log --since={last_tag} | "
    cmd += "sgpt --no-md 'summarize these commit messages into Python/github release notes'"
    notes = c.run(cmd).stdout
    with open("RELEASE_NOTES.md", "w") as stream:
        stream.write(notes)


def read_version():
    # open the pyproject.toml file and read the version number
    with open("pyproject.toml") as stream:
        for line in stream:
            if "version" in line:
                version = line.split("=")[1].strip()
                version = version.replace('"', "")
                return version


@task
def release(c):
    "Tag the current version as a release on Github."
    assert os.path.exists("RELEASE_NOTES.md")
    assert c.run("bump2version --tag patch").ok
    assert c.run("git push --follow-tags").ok
    version = read_version()
    tag = "v" + version
    assert c.run(f"gh release create {tag} -t {tag} --notes-file RELEASE_NOTES.md").ok
    assert c.run(f"rm RELEASE_NOTES.md").ok
    print(f"Release {version} created successfully.")
