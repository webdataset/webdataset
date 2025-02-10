import json
import os
import subprocess
import textwrap
import time
import glob
import re
import typer

app = typer.Typer()


def wrap_long_lines(text, width=80, threshold=120):
    lines = text.split("\n")
    wrapped_lines = []
    for line in lines:
        if len(line) > threshold:
            wrapped_lines.append(textwrap.fill(line, width))
        else:
            wrapped_lines.append(line)
    return "\n".join(wrapped_lines)


FAQ_INTRO = """
# WebDataset FAQ

This is a Frequently Asked Questions file for WebDataset.  It is
automatically generated from selected WebDataset issues using AI.

Since the entries are generated automatically, not all of them may
be correct.  When in doubt, check the original issue.

"""


SUMMARIZE = """
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


def summarize_issue(content):
    result = subprocess.run(
        [
            "sgpt",
            "--no-md",
            SUMMARIZE,
        ],
        input=content.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.decode()


def generate_faq_entries_from_issues():
    """Create FAQ entries from issues."""
    assert os.path.isdir(
        "faqs"
    ), "Please create a directory named 'faqs' before running this task."

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
            "gh issue view {} --repo webdataset/webdataset --json body,title,comments".format(
                issue_number
            ),
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
        combined_content = (
            f"# {issue_title}\n\n{issue_body}\n\n## Comments\n\n{comments}"
        )

        # Pipe the combined content to the summarize function and write the output to a file
        summarized_content = summarize_issue(combined_content)
        with open(output, "w") as f:
            f.write(summarized_content)
        print(summarized_content)
        time.sleep(3)
        print("\n\n")


@app.command()
def genfaq():
    generate_faq_entries_from_issues()
    output = open("FAQ.md", "w")
    output.write(FAQ_INTRO)
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
    os.system("cp FAQ.md docs/FAQ.md")

if __name__ == "__main__":
    app()
