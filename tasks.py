import glob
import os
import re
import shutil
import sys
import tempfile
import textwrap
from invoke import task
import subprocess
import json
import time
from invoke import task
import subprocess
import textwrap


from invoke import task

VENV = "venv"
BIN = f"{VENV}/bin"
PYTHON3 = f"{BIN}/python3"
ACTIVATE = f". {BIN}/activate;"
PIP = f"{BIN}/pip"
PACKAGE = "webdataset"
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
    c.run(f"{BIN}/pip install '.[dev]'")
    print("done")

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
        c.run(f"{ACTIVATE}jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {nb}")
        c.run(f"{ACTIVATE}jupyter nbconvert {nb} --to markdown --output-dir=docs/examples")

@task
def nbrun(c):
    """Run selected notebooks with papermill+parameters; put into ./out."""
    def nbprocess(c, nb, *args, **kwargs):
        """Process one notebook."""
        out_file = f"docs/output/{nb}"
        if not os.path.exists(out_file) or os.path.getmtime(nb) > os.path.getmtime(out_file):
            c.run(f"../venv/bin/python -m papermill -l python {' '.join(args)} {nb} docs/output/_{nb}")
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
def test(c):
    "Run the tests."
    # venv(c)
    c.run(f"{ACTIVATE}{PYTHON3} -m pytest -x tests")


@task
def testwids(c):
    "Run the wids tests."
    c.run(f"{ACTIVATE}{PYTHON3} -m pytest -x tests/test_wids*.py")


@task
def debugtest(c):
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


# @task
# def coverage(c):
#     """Run tests and test coverage."""
#     c.run("coverage run -m pytest && coveragepy-lcov")


@task
def docsserve(c):
    "Serve docs."
    c.run(f"mkdocs serve")


@task
def docspush(c):
    "Push docs to Github pages."
    c.run(f"mkdocs gh-deploy")


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


required_files = """
.github/workflows/pypi.yml
.github/workflows/test.yml
.githooks/pre-push
.gitignore
""".strip().split()


def wrap_long_lines(text, width=80, threshold=120):
    lines = text.split("\n")
    wrapped_lines = []
    for line in lines:
        if len(line) > threshold:
            wrapped_lines.append(textwrap.fill(line, width))
        else:
            wrapped_lines.append(line)
    return "\n".join(wrapped_lines)


faq_intro = """
# WebDataset FAQ

This is a Frequently Asked Questions file for WebDataset.  It is
automatically generated from selected WebDataset issues using AI.

Since the entries are generated automatically, not all of them may
be correct.  When in doubt, check the original issue.

"""


@task
def faqmake(c):
    "Create the FAQ.md file from faqs/*.md"
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


summarize_issue_instructions = """
    - turn this issue report into an FAQ entry if and only if it contains some useful information for users
    - if it does not contain useful information, ONLY return the string N/A
    - the FAQ entry should focus on the single most important part of the issue
    - the FAQ should start with a short Q: and then have an A: that is 1-2 paragraphs long
    - be sure that both Q: and A: start in the first column
    - the very first characters of your answer should be "Q: "
    - you can include 1-2 short code examples
    - use Markdown format to format the output
    - be sure to use ```...``` and `...` consistently for all code
    - be sure your quotes are matching and that you are using proper Markdown formatting
    - YOU MUST USE MARKDOWN FORMAT FOR YOUR OUTPUT
    - DO NOT EVER RETURN CODE BLOCKS WITHOUT SURROUNDING THEM WITH ```...```
"""


def summarize_issue(c, content):
    result = subprocess.run(
        [
            "sgpt",
            "--no-md",
            summarize_issue_instructions,
        ],
        input=content.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.decode()


@task
def faqgen(c):
    """Create FAQ entries from issues."""
    assert os.path.isdir("faqs"), "Please create a directory named 'faqs' before running this task."

    # Fetch all issues (open and closed) from the repository
    s = subprocess.run(
        "gh issue list --repo webdataset/webdataset --label faq --state all --json number",
        stdout=subprocess.PIPE,
        shell=True,
    ).stdout.decode()
    issues = json.loads(s)
    s = subprocess.run(
        "gh issue list --repo webdataset/webdataset --label documentation --state all --json number",
        stdout=subprocess.PIPE,
        shell=True,
    ).stdout.decode()
    issues += json.loads(s)

    # Iterate over each issue
    for issue in issues:
        issue_number = issue["number"]
        output = f"./faqs/{issue_number:04d}.md"
        if os.path.isfile(output):
            continue

        print(f"=== {issue_number} ===\n")

        # Fetch the issue details and comments
        issue_details = subprocess.run(
            "gh issue view {} --repo webdataset/webdataset --json body,title,comments".format(issue_number),
            stdout=subprocess.PIPE,
            shell=True,
        ).stdout.decode()
        issue_details = json.loads(issue_details)

        # Extract the issue title and body
        issue_title = issue_details["title"]
        issue_body = issue_details["body"]

        # Extract the comments
        comments = "\n\n".join(comment["body"] for comment in issue_details["comments"])

        # Combine the issue title, body, and comments
        combined_content = f"# {issue_title}\n\n{issue_body}\n\n## Comments\n\n{comments}"

        # Pipe the combined content to the summarize function and write the output to a file
        summarized_content = summarize_issue(c, combined_content)
        with open(output, "w") as f:
            f.write(summarized_content)
        print(summarized_content)
        time.sleep(3)
        print("\n\n")


summarize_version_instructions = """
Summarize the changes in this git diff in one concise paragraph
suitable for inclusion in a changelog or release notes.
Do not output any Markdown section headers (like ## or ###).
Use Markdown lists to structure the output more cleanly.
Do not simply describe changes ("added xyz file", "updated abc function"), only summarize the intention/meaning of changes.
Leave out any comments related to changes of the maintainer or project status.
Leave out any comments related to VERSION files or version numbers.
Leave out any comments related to README files.
Leave out any comments related to changes formatting or coding style
Be sure to use Markdown conventions to quote code or filenames: `like this`.
Do not include verbiage like "The git diff shows..."
Do not include verbiage like "Updated version to..." or "Reverted version to..." or "updated ... to reflect new version"
Do NOT include comments like "Modified setup.py to update version number".
NO COMMENTS ABOUT VERSIONS OR VERSION NUMBERS, EVER!!!
"""


def summarize_version(commit, prev_commit):
    maxsize = 200000

    diff = subprocess.run(
        f"git log {prev_commit}..{commit} --decorate=short; git diff --stat {prev_commit} {commit}; git diff {prev_commit} {commit} -- '*.py'",
        capture_output=True,
        text=True,
        shell=True,
    ).stdout

    if len(diff) > maxsize:
        print(f"WARNING: diff too large ({len(diff)} bytes), truncating to {maxsize} bytes", file=sys.stderr)

    diff = diff[:maxsize]

    result = subprocess.run(
        ["sgpt", "--no-md", summarize_version_instructions], input=diff, capture_output=True, text=True
    ).stdout

    return result


@task
def versions(ctx, n=200):
    """Summarize the changes in the last n commits."""
    commits = subprocess.run(
        f"git log --pretty=format:'%ai %h %d' -n{n}", shell=True, capture_output=True, text=True
    ).stdout

    print("# Commit Summaries\n")

    commit = "HEAD"

    output_stream = open("VERSIONS.md", "w")

    for line in commits.splitlines():
        if "tag:" in line:
            parts = line.split()
            prev_d, prev_t, prev_z, prev_commit = parts[0], parts[1], parts[2], parts[3]
            message = subprocess.run(
                "git log --format=%B -n 1 {}".format(prev_commit), capture_output=True, text=True, shell=True
            ).stdout.strip()

            summary = summarize_version(commit, prev_commit)

            tag = subprocess.run(
                "git describe --tags {}".format(commit), capture_output=True, text=True, shell=True
            ).stdout.strip()

            prev_tag = subprocess.run(
                "git describe --tags {}".format(prev_commit), capture_output=True, text=True, shell=True
            ).stdout.strip()
            result = f"## Commit: {prev_tag} -> {tag}\n\n"
            result += f"{prev_commit} -> {commit} @ {prev_d} {prev_t} {prev_z}\n\n"
            result += f"{summary}\n"

            print(result)
            output_stream.write(result)

            commit = prev_commit

    output_stream.close()


def get_changes(version):
    """Get the changes for the given version."""
    try:
        changes = summarize_version(version, "last_release")
        return changes
    except:
        print("summarize_changes failed, just using git log")
        return subprocess.run(
            f"git log last_release..{version} --oneline", shell=True, capture_output=True, text=True
        ).stdout


def update_version_numbers_locally(c):
    c.run("bump2version patch")


@task
def release(c):
    "Tag the current version as a release on Github."
    assert c.run("bump2version patch").ok
    tag = "v"+open("VERSION").read().strip()
    print()
    print(f"Creating release {tag}")
    changes = get_changes(tag)
    print()
    print(changes)
    print()
    assert c.run(f"gh release create {tag} -t {tag} --notes-file -", input=changes).ok

    print(f"Release {version} created successfully.")
