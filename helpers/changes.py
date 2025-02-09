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
        print(
            f"WARNING: diff too large ({len(diff)} bytes), truncating to {maxsize} bytes",
            file=sys.stderr,
        )

    diff = diff[:maxsize]

    result = subprocess.run(
        ["sgpt", "--no-md", summarize_version_instructions],
        input=diff,
        capture_output=True,
        text=True,
    ).stdout

    return result


@task
def versions(ctx, n=200):
    """Summarize the changes in the last n commits."""
    commits = subprocess.run(
        f"git log --pretty=format:'%ai %h %d' -n{n}",
        shell=True,
        capture_output=True,
        text=True,
    ).stdout

    print("# Commit Summaries\n")

    commit = "HEAD"

    output_stream = open("VERSIONS.md", "w")

    for line in commits.splitlines():
        if "tag:" in line:
            parts = line.split()
            prev_d, prev_t, prev_z, prev_commit = parts[0], parts[1], parts[2], parts[3]
            message = subprocess.run(
                "git log --format=%B -n 1 {}".format(prev_commit),
                capture_output=True,
                text=True,
                shell=True,
            ).stdout.strip()

            summary = summarize_version(commit, prev_commit)

            tag = subprocess.run(
                "git describe --tags {}".format(commit),
                capture_output=True,
                text=True,
                shell=True,
            ).stdout.strip()

            prev_tag = subprocess.run(
                "git describe --tags {}".format(prev_commit),
                capture_output=True,
                text=True,
                shell=True,
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
            f"git log last_release..{version} --oneline",
            shell=True,
            capture_output=True,
            text=True,
        ).stdout
