"""Small, standard-library-only helpers for the OOOBrush Release workflow."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import Request, urlopen
import zipfile


TARGETS = {
    "x86_64-pc-windows-msvc": ".zip",
    "x86_64-unknown-linux-gnu": ".tar.gz",
    "aarch64-apple-darwin": ".tar.gz",
}
TAG_PATTERN = re.compile(
    r"ooo-v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
)


def validate_tag(tag):
    match = TAG_PATTERN.fullmatch(tag)
    if not match:
        raise ValueError("Expected ooo-vMAJOR.MINOR.PATCH, optionally with -rc.1 etc.")
    suffix = match.group(4)
    if suffix and any(part.isdigit() and len(part) > 1 and part[0] == "0"
                      for part in suffix.split(".")):
        raise ValueError("Numeric prerelease identifiers cannot have leading zeros")
    return suffix is not None


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def resolve_build(env):
    event, ref = env["GITHUB_EVENT_NAME"], env["GITHUB_REF"]
    publishing = event == "workflow_dispatch" or (
        event == "push" and ref.startswith("refs/tags/ooo-v")
    )
    tag = ""
    source = env["GITHUB_SHA"]
    if publishing:
        tag = env.get("INPUT_RELEASE_TAG", "") if event == "workflow_dispatch" else ref[10:]
        validate_tag(tag)
        tag_ref = f"refs/tags/{tag}"
        exists = subprocess.run(["git", "show-ref", "--verify", "--quiet", tag_ref])
        if exists.returncode == 0:
            source = tag_ref
        elif exists.returncode != 1:
            raise RuntimeError("Failed to inspect release tag")
        elif event != "workflow_dispatch":
            raise RuntimeError("The pushed release tag is missing from checkout")
    commit = git("rev-parse", "--verify", f"{source}^{{commit}}")
    return {"commit": commit, "tag": tag, "publishing": str(publishing).lower()}


def sha256(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return sha256(stream)


def validate_archives(directory, commit):
    """Require all three packages, matching commits, inner hashes and Unix modes."""
    expected = {f"OOOBrush-{target}{ext}" for target, ext in TARGETS.items()}
    if {path.name for path in directory.iterdir()} - {"SHA256SUMS"} != expected:
        raise ValueError("Expected exactly the three platform archives")
    summaries = []
    for target, ext in TARGETS.items():
        path = directory / f"OOOBrush-{target}{ext}"
        windows = ext == ".zip"
        with zipfile.ZipFile(path) if windows else tarfile.open(path, "r:gz") as archive:
            entries = archive.infolist() if windows else archive.getmembers()
            members = {}
            for entry in entries:
                is_file = not entry.is_dir() if windows else entry.isfile()
                if not is_file:
                    continue
                name = (entry.filename if windows else entry.name).removeprefix("./")
                if name in members:
                    raise ValueError(f"Duplicate archive member: {name}")
                members[name] = entry
            suffix = ".exe" if windows else ""
            binaries = {f"brush{suffix}", f"brush-cli{suffix}"}
            required = binaries | {"README.md", "LICENSE", "BUILDINFO.json", "SHA256SUMS"}
            if set(members) != required:
                raise ValueError(f"Incomplete/unexpected contents in {path.name}")

            def open_member(name):
                return archive.open(members[name]) if windows else archive.extractfile(members[name])

            with open_member("BUILDINFO.json") as stream:
                info = json.load(stream)
            if info.get("commit") != commit or info.get("target") != target:
                raise ValueError(f"Source commit or target mismatch in {path.name}")
            versions = info.get("versions", {})
            if set(versions) != {"brush", "brush-cli"} or not all(
                isinstance(version, str) and version.strip() for version in versions.values()
            ):
                raise ValueError(f"Missing binary versions in {path.name}")
            with open_member("SHA256SUMS") as stream:
                lines = stream.read().decode("utf-8-sig").splitlines()
            checksums = {}
            for line in lines:
                match = re.fullmatch(r"([0-9a-fA-F]{64})  (.+)", line)
                if not match or match[2] in checksums:
                    raise ValueError(f"Invalid internal checksum in {path.name}")
                checksums[match[2]] = match[1].lower()
            if set(checksums) != required - {"SHA256SUMS"}:
                raise ValueError(f"Incomplete internal checksums in {path.name}")
            for name, expected_hash in checksums.items():
                with open_member(name) as stream:
                    if sha256(stream) != expected_hash:
                        raise ValueError(f"Checksum mismatch: {path.name}/{name}")
            if not windows and any(not members[name].mode & 0o111 for name in binaries):
                raise ValueError(f"Missing Unix executable permissions in {path.name}")
            summaries.append((target, versions))
    return summaries


def read_release(repo, tag):
    # The by-tag endpoint only promises published releases. The authenticated
    # list endpoint includes drafts and supports resuming interrupted uploads.
    page = 1
    while True:
        request = Request(
            f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}",
            headers={"Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                     "Accept": "application/vnd.github+json", "User-Agent": "OOOBrush-release"},
        )
        with urlopen(request, timeout=60) as response:
            releases = json.load(response)
        for release in releases:
            if release["tag_name"] == tag:
                return release
        if len(releases) < 100:
            return None
        page += 1


def remote_tag_commit(tag):
    ref = f"refs/tags/{tag}"
    lines = git("ls-remote", "--tags", "origin", ref, f"{ref}^{{}}")
    refs = dict(line.split()[::-1] for line in lines.splitlines())
    return refs.get(f"{ref}^{{}}", refs.get(ref))


def gh(repo, *args):
    subprocess.run(["gh", "release", *args, "--repo", repo], check=True)


def require_owned_draft(release, commit):
    if not release or not release.get("draft"):
        raise ValueError("Release is already public or the expected draft is missing")
    marker = f"<!-- ooobrush-release commit={commit} -->"
    if release.get("target_commitish") != commit or marker not in (release.get("body") or ""):
        raise ValueError("Existing draft belongs to a different commit/workflow")


def publish(directory, repo, tag, commit):
    prerelease = validate_tag(tag)
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ValueError("Expected a full source commit SHA")
    summaries = validate_archives(directory, commit)
    release = read_release(repo, tag)
    if release is not None:
        require_owned_draft(release, commit)
    remote_commit = remote_tag_commit(tag)
    if remote_commit is not None and remote_commit != commit:
        raise ValueError("Remote release tag no longer points to the build commit")
    archives = sorted(directory.glob("OOOBrush-*"))
    checksums = directory / "SHA256SUMS"
    checksums.write_text("".join(f"{file_hash(path)}  {path.name}\n" for path in archives), encoding="utf-8")
    notes = directory.parent / "release-notes.md"
    version_lines = "\n".join(
        f"| `{target}` | `{versions['brush-cli']}` | `{versions['brush']}` |"
        for target, versions in summaries
    )
    notes.write_text(
        f"# OOOBrush {tag}\n\nSource commit: `{commit}`\n\n"
        "| Target | CLI version | Application version |\n|---|---|---|\n"
        f"{version_lines}\n\n"
        "Download the ZIP (Windows x64) or tar.gz (Linux x64/macOS ARM64). "
        "Each contains both binaries, README, LICENSE, BUILDINFO.json and internal checksums. "
        "Verify the downloaded archives with the attached SHA256SUMS.\n\n"
        "For OOOSplat, use brush-cli (brush-cli.exe on Windows). "
        "Set RUST_LOG=info and RUST_BACKTRACE=1 for diagnostics; "
        "training uses --total-train-iters. Builds are unsigned; macOS is not notarized. "
        "Hosted-runner smoke tests do not validate GPU training.\n\n"
        f"<!-- ooobrush-release commit={commit} -->\n", encoding="utf-8",
    )
    if release is None:
        # GitHub may defer creating a nonexistent tag until a draft is
        # published. Create it explicitly at the verified build SHA first.
        if remote_commit is None:
            subprocess.run([
                "gh", "api", "--method", "POST", f"repos/{repo}/git/refs",
                "-f", f"ref=refs/tags/{tag}", "-f", f"sha={commit}",
            ], check=True)
        args = ["create", tag, "--target", commit, "--draft", "--title", f"OOOBrush {tag}",
                "--notes-file", str(notes), "--verify-tag"]
        if prerelease:
            args += ["--prerelease"]
        gh(repo, *args)
    require_owned_draft(read_release(repo, tag), commit)
    gh(repo, "upload", tag, *(str(path) for path in [*archives, checksums]), "--clobber")
    # Verify actual downloaded bytes while still a draft, including on older
    # GitHub installations without Release asset digest metadata.
    with tempfile.TemporaryDirectory(prefix="ooobrush-release-") as temporary:
        args = ["download", tag, "--dir", temporary]
        for path in [*archives, checksums]:
            args += ["--pattern", path.name]
        gh(repo, *args)
        for path in [*archives, checksums]:
            downloaded = Path(temporary) / path.name
            if not downloaded.is_file() or file_hash(downloaded) != file_hash(path):
                raise ValueError(f"Uploaded asset verification failed: {path.name}")
    current = read_release(repo, tag)
    require_owned_draft(current, commit)
    if {asset["name"] for asset in current.get("assets", [])} != {path.name for path in [*archives, checksums]}:
        raise ValueError("Draft does not contain exactly the four expected attachments")
    if remote_tag_commit(tag) != commit:
        raise ValueError("Remote release tag changed before publication")
    gh(repo, "edit", tag, "--draft=false", f"--prerelease={str(prerelease).lower()}",
       f"--latest={str(not prerelease).lower()}", "--notes-file", str(notes))
    final = read_release(repo, tag)
    if not final or final.get("draft") or final.get("prerelease") != prerelease:
        raise ValueError("Release publication could not be confirmed")
    print(f"Published {final['html_url']} from {commit}")


if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        values = resolve_build(os.environ)
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write("".join(f"{key}={value}\n" for key, value in values.items()))
        print(f"Build commit: {values['commit']}; release tag: {values['tag'] or '(checks only)'}")
    elif len(sys.argv) == 3 and sys.argv[1] == "publish":
        publish(Path(sys.argv[2]), os.environ["GITHUB_REPOSITORY"],
                os.environ["RELEASE_TAG"], os.environ["RELEASE_COMMIT"])
    else:
        sys.exit("Usage: cli_release.py prepare | publish <artifact-directory>")
