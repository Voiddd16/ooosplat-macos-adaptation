"""Release regression tests use only local files and mocked GitHub calls."""

import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
import zipfile

import cli_release as release


COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40
TAG = "ooo-v1.0.0"
REPO = "ooolabdev/OOOBrush"


def make_packages(directory, commit=COMMIT, bad_mode=False, bad_checksum=False):
    for target, ext in release.TARGETS.items():
        suffix = ".exe" if ext == ".zip" else ""
        files = {
            "README.md": b"Usage", "LICENSE": b"MIT",
            f"brush{suffix}": b"application", f"brush-cli{suffix}": b"CLI",
            "BUILDINFO.json": json.dumps({
                "commit": commit, "target": target,
                "versions": {"brush": "brush 1.0.0", "brush-cli": "brush-cli 1.0.0"},
            }).encode(),
        }
        files["SHA256SUMS"] = "".join(
            f"{hashlib.sha256(data).hexdigest()}  {name}\n"
            for name, data in sorted(files.items())
        ).encode()
        if bad_checksum:
            files[f"brush-cli{suffix}"] = b"corrupted binary"
        path = directory / f"OOOBrush-{target}{ext}"
        if ext == ".zip":
            with zipfile.ZipFile(path, "w") as archive:
                for name, data in files.items():
                    archive.writestr(name, data)
        else:
            with tarfile.open(path, "w:gz") as archive:
                for name, data in files.items():
                    member = tarfile.TarInfo(f"./{name}")
                    member.size = len(data)
                    member.mode = 0o644 if bad_mode or not name.startswith("brush") else 0o755
                    archive.addfile(member, io.BytesIO(data))


class ResolveTests(unittest.TestCase):
    def env(self, event="push", ref="refs/heads/main", **extra):
        return {"GITHUB_EVENT_NAME": event, "GITHUB_REF": ref, "GITHUB_SHA": COMMIT, **extra}

    def test_pr_and_main_only_build(self):
        for event, ref in [("push", "refs/heads/main"), ("pull_request", "refs/pull/1/merge")]:
            with self.subTest(event=event), patch.object(release, "git", return_value=COMMIT), patch.object(release.subprocess, "run") as run:
                self.assertEqual(release.resolve_build(self.env(event, ref)), {
                    "commit": COMMIT, "tag": "", "publishing": "false",
                })
                run.assert_not_called()

    def test_pushed_and_existing_manual_tag_resolve_commit(self):
        for env in [self.env(ref=f"refs/tags/{TAG}"), self.env("workflow_dispatch", INPUT_RELEASE_TAG=TAG)]:
            with self.subTest(env=env), patch.object(release.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), patch.object(release, "git", return_value=OTHER_COMMIT) as git:
                result = release.resolve_build(env)
                self.assertEqual(result, {"commit": OTHER_COMMIT, "tag": TAG, "publishing": "true"})
                git.assert_called_once_with("rev-parse", "--verify", f"refs/tags/{TAG}^{{commit}}")

    def test_new_manual_tag_uses_selected_run_commit(self):
        with patch.object(release.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)), patch.object(release, "git", return_value=COMMIT) as git:
            result = release.resolve_build(self.env("workflow_dispatch", INPUT_RELEASE_TAG=TAG))
            self.assertEqual(result["commit"], COMMIT)
            git.assert_called_once_with("rev-parse", "--verify", f"{COMMIT}^{{commit}}")

    def test_missing_push_tag_and_git_failure_are_errors(self):
        for code in [1, 128]:
            with self.subTest(code=code), patch.object(release.subprocess, "run", return_value=subprocess.CompletedProcess([], code)), self.assertRaises(RuntimeError):
                release.resolve_build(self.env(ref=f"refs/tags/{TAG}"))

    def test_tag_validation_and_prerelease(self):
        self.assertFalse(release.validate_tag(TAG))
        self.assertTrue(release.validate_tag("ooo-v2.3.4-rc.1"))
        for tag in ["v1.0.0", "ooo-v1.2", "ooo-v01.2.3", "ooo-v1.2.3-rc.01", "ooo-v1.2.3\ncommit=bad", "ooo-v1.2.3;echo bad", "", "ooo-v1.2.3+build"]:
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                release.validate_tag(tag)

    def test_annotated_tag_uses_peeled_commit(self):
        with patch.object(release, "git", return_value=f"{OTHER_COMMIT}\trefs/tags/{TAG}\n{COMMIT}\trefs/tags/{TAG}^{{}}"):
            self.assertEqual(release.remote_tag_commit(TAG), COMMIT)


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "artifacts"
        self.directory.mkdir()
        make_packages(self.directory)
        self.current = None
        self.remote = COMMIT
        self.commands = []
        self.corrupt_download = False
        self.fail_upload = False
        self.change_tag_after_download = False
        self.extra_asset = False

    def draft(self, commit=COMMIT, tag=TAG):
        return {"draft": True, "target_commitish": commit,
                "body": f"<!-- ooobrush-release commit={commit} -->",
                "assets": [], "prerelease": release.validate_tag(tag),
                "html_url": f"https://github.com/{REPO}/releases/tag/{tag}"}

    def fake_gh(self, repo, *args):
        self.assertEqual(repo, REPO)
        self.commands.append(args)
        if args[0] == "create":
            self.current = self.draft(tag=args[1])
        elif args[0] == "upload":
            if self.fail_upload:
                raise subprocess.CalledProcessError(1, ["gh", *args])
            self.current["assets"] = [{"name": path.name} for path in self.directory.iterdir()]
            if self.extra_asset:
                self.current["assets"].append({"name": "unrelated.zip"})
        elif args[0] == "download":
            destination = Path(args[args.index("--dir") + 1])
            for path in self.directory.iterdir():
                (destination / path.name).write_bytes(b"corrupted" if self.corrupt_download else path.read_bytes())
            if self.change_tag_after_download:
                self.remote = OTHER_COMMIT
        elif args[0] == "edit":
            self.assertIn("--draft=false", args)
            self.current["draft"] = False

    def run_publish(self, tag=TAG):
        def create_tag(args, **kwargs):
            self.assertEqual(args[:5], ["gh", "api", "--method", "POST", f"repos/{REPO}/git/refs"])
            self.assertIn(f"sha={COMMIT}", args)
            self.remote = COMMIT
        with patch.object(release, "read_release", side_effect=lambda *_: copy.deepcopy(self.current)), patch.object(release, "remote_tag_commit", side_effect=lambda _: self.remote), patch.object(release, "gh", side_effect=self.fake_gh), patch.object(release.subprocess, "run", side_effect=create_tag) as create:
            release.publish(self.directory, REPO, tag, COMMIT)
            return create

    def assert_not_published(self):
        self.assertFalse(any(args[0] == "edit" for args in self.commands))

    def test_three_packages_inner_hashes_and_unix_permissions(self):
        self.assertEqual(len(release.validate_archives(self.directory, COMMIT)), 3)
        for invalid in ["missing", "commit", "permissions", "checksum"]:
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                make_packages(directory, commit=OTHER_COMMIT if invalid == "commit" else COMMIT,
                              bad_mode=invalid == "permissions", bad_checksum=invalid == "checksum")
                if invalid == "missing":
                    next(directory.glob("*.zip")).unlink()
                with self.assertRaises(ValueError):
                    release.validate_archives(directory, COMMIT)

    def test_new_draft_verified_before_publication(self):
        create = self.run_publish()
        create.assert_not_called()
        self.assertEqual([args[0] for args in self.commands], ["create", "upload", "download", "edit"])
        self.assertIn("--verify-tag", self.commands[0])
        self.assertIn("--latest=true", self.commands[-1])
        self.assertIn("--prerelease=false", self.commands[-1])
        checksums = (self.directory / "SHA256SUMS").read_text()
        self.assertEqual(len(checksums.splitlines()), 3)
        notes = (self.directory.parent / "release-notes.md").read_text()
        self.assertIn(COMMIT, notes)
        self.assertIn("brush-cli 1.0.0", notes)

    def test_missing_manual_tag_is_created_at_build_sha(self):
        self.remote = None
        create = self.run_publish()
        create.assert_called_once()

    def test_owned_draft_can_resume(self):
        self.current = self.draft()
        self.run_publish()
        self.assertEqual([args[0] for args in self.commands], ["upload", "download", "edit"])

    def test_public_or_other_commit_draft_is_not_overwritten(self):
        for current in [dict(self.draft(), draft=False), self.draft(OTHER_COMMIT), dict(self.draft(), body="unrelated")]:
            with self.subTest(current=current):
                self.current = current
                with self.assertRaises(ValueError):
                    self.run_publish()
                self.assertEqual(self.commands, [])

    def test_changed_remote_tag_stops_publication(self):
        self.remote = OTHER_COMMIT
        with self.assertRaises(ValueError):
            self.run_publish()
        self.assertEqual(self.commands, [])

    def test_tag_changed_during_upload_keeps_draft(self):
        self.change_tag_after_download = True
        with self.assertRaisesRegex(ValueError, "changed before publication"):
            self.run_publish()
        self.assertTrue(self.current["draft"])
        self.assert_not_published()

    def test_missing_platform_never_creates_a_draft(self):
        next(self.directory.glob("*.zip")).unlink()
        with self.assertRaises(ValueError):
            self.run_publish()
        self.assertEqual(self.commands, [])

    def test_unexpected_draft_attachment_blocks_publication(self):
        self.extra_asset = True
        with self.assertRaisesRegex(ValueError, "exactly the four expected attachments"):
            self.run_publish()
        self.assertTrue(self.current["draft"])
        self.assert_not_published()

    def test_upload_or_download_verification_failure_keeps_draft(self):
        for failure in ["upload", "download"]:
            with self.subTest(failure=failure):
                self.current = None
                self.commands = []
                self.fail_upload = failure == "upload"
                self.corrupt_download = failure == "download"
                with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                    self.run_publish()
                self.assertTrue(self.current["draft"])
                self.assert_not_published()

    def test_prerelease_is_not_latest(self):
        self.run_publish("ooo-v1.0.0-rc.1")
        self.assertIn("--prerelease", self.commands[0])
        self.assertIn("--prerelease=true", self.commands[-1])
        self.assertIn("--latest=false", self.commands[-1])

    def test_release_list_includes_drafts_and_paginates(self):
        draft = dict(self.draft(), tag_name=TAG)
        pages = [io.BytesIO(json.dumps([{"tag_name": f"v0.0.{i}"} for i in range(100)]).encode()),
                 io.BytesIO(json.dumps([draft]).encode())]
        with patch.dict(release.os.environ, {"GH_TOKEN": "test-token"}), patch.object(release, "urlopen", side_effect=pages) as get:
            self.assertEqual(release.read_release(REPO, TAG), draft)
            self.assertEqual(get.call_count, 2)
            self.assertTrue(get.call_args.args[0].full_url.endswith("page=2"))
        with patch.dict(release.os.environ, {"GH_TOKEN": "test-token"}), patch.object(release, "urlopen", return_value=io.BytesIO(b"[]")):
            self.assertIsNone(release.read_release(REPO, TAG))

    def test_api_errors_are_not_treated_as_missing_release(self):
        for code in [404, 403, 500]:
            with self.subTest(code=code), patch.dict(release.os.environ, {"GH_TOKEN": "test-token"}), patch.object(release, "urlopen", side_effect=HTTPError("https://api.github.com/", code, "test", {}, None)):
                with self.assertRaises(HTTPError):
                    release.read_release(REPO, TAG)


if __name__ == "__main__":
    unittest.main()
