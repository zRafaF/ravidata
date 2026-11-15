"""Update the published dataset (Hugging Face) from the local copy in data/ravidata, without rebuilding it.

    uv run python scripts/hf_release.py status          # read-only: what differs from Hugging Face, which tags exist
    uv run python scripts/hf_release.py patch           # apply hf_dataset.SYNC_CORRECTIONS + card to data/ravidata
    uv run python scripts/hf_release.py upload [--yes]  # tag the old state, upload the changes, tag the new version

`make hf-status`, `make hf-patch`, `make hf-release` (= patch + upload --yes). Uploading needs a write token
(`uv run hf auth login`). Without --yes, `upload` only prints what it would do.

What a release contains (RELEASE below): the metadata files (every take.json and the card, README.md) and the derived
folders the analysis reads (fused poses of the maps, depth of the still takes). Files that are already identical on
Hugging Face are skipped. Before the first change the current state of the dataset is tagged PREVIOUS, so results made
with it stay reproducible (`hf download ... --revision v1.0`); at the end the new state is tagged
hf_dataset.DATASET_VERSION.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hf_dataset as H  # noqa: E402  (card text, sync corrections, takes)

DATA = ROOT / "data" / "ravidata"
REPO = "DoninhaD/ravidata"
PREVIOUS = "v1.0"
FOLDERS = ["derived/maps-v1/fused", "derived/static-depth-v1"]      # inputs of the analysis made on a PC


def meta_files() -> list[str]:
    return ["README.md"] + [f"takes/{t}/take.json" for t in H.TAKES if (DATA / "takes" / t / "take.json").exists()]


def git_blob_sha1(path: Path) -> str:
    """Hugging Face's `blob_id` of a small (non-LFS) file: git's blob hash."""
    b = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def _local_ids(path: Path) -> tuple[str, str]:
    """(git blob sha1, sha256) of a local file: Hugging Face reports the first for small files, the second for LFS."""
    b = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest(), hashlib.sha256(b).hexdigest()


def local_tree(prefix: str) -> dict[str, int]:
    base = DATA / prefix
    return {f"{prefix}/{p.relative_to(base).as_posix()}": p.stat().st_size for p in sorted(base.rglob("*")) if p.is_file()}


# ------------------------------------------------------------------------------------------------------------- patch
def patch() -> list[str]:
    """Bring the local metadata to DATASET_VERSION: sync corrections, version field, card. Idempotent."""
    if not (DATA / "takes").exists():
        raise SystemExit(f"no dataset in {DATA}: run `make data` first")
    changed = []
    for name in H.TAKES:
        tj = DATA / "takes" / name / "take.json"
        if not tj.exists():
            continue
        old = tj.read_text(encoding="utf-8")
        t = json.loads(old)
        t = {"dataset_version": H.DATASET_VERSION, **{k: v for k, v in t.items() if k != "dataset_version"}}
        t["sync"] = H.correct_sync(name, t["sync"])
        new = json.dumps(t, indent=1, default=str)
        if new != old:
            tj.write_text(new, encoding="utf-8")
            changed.append(str(tj.relative_to(DATA)))
    card = H.render_card(DATA)
    readme = DATA / "README.md"
    if not readme.exists() or readme.read_text(encoding="utf-8") != card:
        readme.write_text(card, encoding="utf-8")
        changed.append("README.md")
    check_words([DATA / f for f in meta_files()])
    for c in changed:
        print(f"patched {c}")
    if not changed:
        print(f"data/ravidata metadata already at {H.DATASET_VERSION}")
    return changed


def check_words(paths):
    """Same rule as hf_dataset.neutralise: published text names no company or internal tool."""
    import re
    words = ["neo" + "senti", "ns" + "-radar", "ns" + "radar", "bia" + "-radar", "EXP-0", r"\bD-0\d"]
    bad = re.compile("|".join(words), re.I)
    for p in paths:
        m = bad.search(p.read_text(encoding="utf-8"))
        if m:
            raise SystemExit(f"{p}: '{m.group(0)}' must not be published")


# ------------------------------------------------------------------------------------------------- remote comparison
def api_or_exit():
    from huggingface_hub import HfApi
    return HfApi()


def remote_state(api, repo: str):
    """(tags, {path: blob_id} of the metadata files, {folder: {path: size}})."""
    tags = [t.name for t in api.list_repo_refs(repo, repo_type="dataset").tags]
    info = api.get_paths_info(repo, meta_files(), repo_type="dataset")
    blobs = {i.path: getattr(i, "blob_id", None) for i in info}
    trees = {}
    for f in FOLDERS:
        try:
            trees[f] = {e.path: (e.lfs.sha256 if getattr(e, "lfs", None) else e.blob_id)
                        for e in api.list_repo_tree(repo, path_in_repo=f, recursive=True, repo_type="dataset")
                        if getattr(e, "blob_id", None)}
        except Exception:                                  # folder not on the hub yet
            trees[f] = {}
    return tags, blobs, trees


def plan(api, repo: str):
    tags, blobs, trees = remote_state(api, repo)
    meta = [f for f in meta_files() if blobs.get(f) != git_blob_sha1(DATA / f)]
    folders, missing = {}, []
    for f in FOLDERS:
        loc = local_tree(f) if (DATA / f).exists() else {}
        if not loc:
            missing.append(f)
            continue
        diff = [p for p in loc if trees[f].get(p) not in (None, *_local_ids(DATA / p))]
        diff += [p for p in loc if p not in trees[f] and p not in diff]
        if diff:
            folders[f] = (len(diff), sum(loc[p] for p in diff))
    return tags, meta, folders, missing


def status(repo: str):
    api = api_or_exit()
    loc_ver = {t: json.loads((DATA / "takes" / t / "take.json").read_text(encoding="utf-8")).get("dataset_version", "v1.0")
               for t in H.TAKES if (DATA / "takes" / t / "take.json").exists()}
    print(f"local data/ravidata: take.json versions {sorted(set(loc_ver.values()))} (code expects {H.DATASET_VERSION})")
    tags, meta, folders, missing = plan(api, repo)
    print(f"hub {repo}: tags {tags or 'none'}")
    print("  metadata to upload: " + (", ".join(meta) if meta else "none (identical)"))
    for f, (n, b) in folders.items():
        print(f"  {f}: {n} files, {b / 1e6:.0f} MB to upload")
    for f in missing:
        print(f"  ! {f}: not in data/ravidata, cannot be compared (the release needs it)")
    if not folders and not missing:
        print("  derived folders: identical")
    if any(v != H.DATASET_VERSION for v in loc_ver.values()):
        print("next: `make hf-patch` (local metadata), then `make hf-release`")


# --------------------------------------------------------------------------------------------------------- upload
def upload(repo: str, yes: bool):
    from huggingface_hub import CommitOperationAdd
    api = api_or_exit()
    if yes:
        try:
            print(f"logged in as {api.whoami()['name']}")
        except Exception:
            raise SystemExit("not logged in: run `uv run hf auth login` with a write token")
    tags, meta, folders, missing = plan(api, repo)
    if missing:
        raise SystemExit("not in data/ravidata: " + ", ".join(missing) + " (the files the published analysis used; "
                         "copy them back, or download them with `make data-all` if they are already on the hub)")
    new = H.DATASET_VERSION
    steps = []
    if (meta or folders) and PREVIOUS not in tags:
        steps.append(("tag", PREVIOUS, f"before {new}: the dataset as used by results/analysis-v1"))
    if meta:
        steps.append(("meta", meta, f"{new}: " + "; ".join(_why(meta))))
    for f in folders:
        steps.append(("folder", f, f"{new}: {f} (the files used by results/analysis-v2)"))
    if new not in tags:
        steps.append(("tag", new, "indoor_course clock fix; fused map poses and static-take depth added"))
    if not steps:
        print("nothing to do: the hub already has this release")
        return
    for s in steps:
        print(("would " if not yes else "") + {"tag": "tag", "meta": "upload", "folder": "upload folder"}[s[0]],
              s[1], "-", s[2])
    if not yes:
        print("(dry run: add --yes, or `make hf-release`)")
        return
    for kind, what, msg in steps:
        if kind == "tag":
            api.create_tag(repo, tag=what, tag_message=msg, repo_type="dataset")
        elif kind == "meta":
            api.create_commit(repo, repo_type="dataset", commit_message=msg,
                              operations=[CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(DATA / f)) for f in what])
        else:
            api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(DATA / what), path_in_repo=what,
                              commit_message=msg)
        print(f"done: {kind} {what if isinstance(what, str) else len(what)}")
    _, meta2, folders2, _ = plan(api, repo)
    print("check: hub identical to data/ravidata" if not meta2 and not folders2 else
          f"check FAILED: still different: {meta2} {list(folders2)}")
    print(f"https://huggingface.co/datasets/{repo}")


def _why(meta):
    out = []
    if any(f.endswith("take.json") for f in meta):
        out.append("take.json: dataset_version; " + ", ".join(
            f"{n} {c['field']} {c['new']}" for n, c in H.SYNC_CORRECTIONS.items() if "new" in c))
    if "README.md" in meta:
        out.append("card")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["status", "patch", "upload"])
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--yes", action="store_true", help="upload: really do it (otherwise a dry run)")
    a = ap.parse_args(argv)
    if a.cmd == "patch":
        patch()
    elif a.cmd == "status":
        status(a.repo)
    else:
        upload(a.repo, a.yes)


if __name__ == "__main__":
    main()
