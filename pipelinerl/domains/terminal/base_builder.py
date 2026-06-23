"""Build per-domain base rootfs images for the terminal domain.

TMax tasks are deltas on a shared per-domain base (``base_<domain>.sif`` in the
upstream repo): the published ``container_def`` assumes domain tools (rust,
ffmpeg, ...) already exist. We reproduce those bases once as proot rootfs
directories, clean them, and snapshot them on shared storage so each rollout
only pays a fast ``cp -a`` reset. See ``TMAX_ENV_RECIPE.md``.

Usage:
    python -m pipelinerl.domains.terminal.base_builder \
        --defs-dir /path/to/tmax/rl_data/containers \
        --out-dir  /mnt/llmd/terminal_bases \
        --proot    /usr/local/bin/proot
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

from .proot_env import _BIND_PATHS, _CONTAINER_ENV, _post_body

logger = logging.getLogger(__name__)

UBUNTU_ROOTFS_URL = (
    "https://images.linuxcontainers.org/images/ubuntu/jammy/amd64/default"
)
DOMAINS = (
    "security", "software_engineering", "file_operations", "data_querying",
    "data_science", "debugging", "scientific_computing", "data_processing",
    "system_administration", "intricate",
)
#: Packages every task assumes; baking them in avoids a per-task apt download.
_COMMON_BOOTSTRAP = "apt-get update && apt-get install -y python3 python3-pip && pip3 install pytest"
_CLEAN = (
    "apt-get clean; rm -rf /var/lib/apt/lists/* /usr/share/doc/* "
    "/usr/share/man/* /var/log/* /tmp/* 2>/dev/null; true"
)


def _latest_ubuntu_tarball() -> str:
    import urllib.request

    with urllib.request.urlopen(f"{UBUNTU_ROOTFS_URL}/") as r:
        listing = r.read().decode()
    import re

    stamps = sorted(re.findall(r"\d{8}_\d{2}:\d{2}", listing))
    if not stamps:
        raise RuntimeError("could not find an ubuntu rootfs build timestamp")
    return f"{UBUNTU_ROOTFS_URL}/{stamps[-1]}/rootfs.tar.xz"


def bootstrap_ubuntu_rootfs(dest: Path, proot_bin: str) -> Path:
    """Download and extract a base Ubuntu rootfs (one-time)."""
    import urllib.request

    dest.mkdir(parents=True, exist_ok=True)
    tarball = dest / "rootfs.tar.xz"
    if not tarball.exists():
        url = _latest_ubuntu_tarball()
        logger.info("downloading ubuntu rootfs from %s", url)
        urllib.request.urlretrieve(url, tarball)
    rootfs = dest / "rootfs"
    if not rootfs.exists():
        rootfs.mkdir()
        subprocess.run(
            [proot_bin, "-0", "tar", "-C", str(rootfs), "--warning=no-all", "-xf", str(tarball)],
            check=True,
        )
    return rootfs


def _proot_exec(proot_bin: str, rootfs: Path, script: str, cwd: str = "/root", timeout: float = 3600) -> int:
    argv = [proot_bin, "-0", "-r", str(rootfs)]
    for b in _BIND_PATHS:
        if Path(b).exists():
            argv += ["-b", b]
    argv += ["-w", cwd, "--kill-on-exit", "/usr/bin/env", "-i", *_CONTAINER_ENV, "/bin/bash", "-c", script]
    proc = subprocess.run(argv, timeout=timeout)
    return proc.returncode


def build_base(
    def_path: Path,
    ubuntu_rootfs: Path,
    out_dir: Path,
    proot_bin: str,
    nameserver: str,
    force: bool = False,
) -> bool:
    """Build one per-domain base rootfs from a ``base_<domain>.def``."""
    name = def_path.stem  # e.g. base_security
    out = out_dir / name
    if out.exists() and not force:
        logger.info("skip %s (exists)", name)
        return True
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)

    t0 = time.perf_counter()
    subprocess.run(["cp", "-a", str(ubuntu_rootfs), str(out)], check=True)
    resolv = out / "etc/resolv.conf"
    resolv.unlink(missing_ok=True)
    resolv.write_text(f"nameserver {nameserver}\noptions ndots:0\n")

    post = _post_body(def_path.read_text())
    script = "\n".join([_COMMON_BOOTSTRAP, post, _CLEAN])
    rc = _proot_exec(proot_bin, out, script)
    if rc != 0:
        logger.error("FAILED %s (rc=%d)", name, rc)
        shutil.rmtree(out, ignore_errors=True)
        return False
    logger.info("built %s in %.1fs", name, time.perf_counter() - t0)
    return True


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--defs-dir", required=True, help="dir with base_<domain>.def files")
    p.add_argument("--out-dir", required=True, help="output dir for base rootfs trees")
    p.add_argument("--proot", default="proot", help="path to the proot binary")
    p.add_argument("--ubuntu-cache", default=None, help="dir to cache the base ubuntu rootfs")
    p.add_argument("--nameserver", default="10.150.0.10")
    p.add_argument("--domains", nargs="*", default=None, help="subset of domains to build")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.ubuntu_cache) if args.ubuntu_cache else out_dir / "_ubuntu"
    ubuntu_rootfs = bootstrap_ubuntu_rootfs(cache, args.proot)

    defs_dir = Path(args.defs_dir)
    domains = args.domains or list(DOMAINS)
    ok = 0
    for domain in domains:
        def_path = defs_dir / f"base_{domain}.def"
        if not def_path.exists():
            logger.warning("missing def %s, skipping", def_path)
            continue
        if build_base(def_path, ubuntu_rootfs, out_dir, args.proot, args.nameserver, args.force):
            ok += 1
    logger.info("built %d/%d base images", ok, len(domains))
    return 0 if ok == len(domains) else 1


if __name__ == "__main__":
    sys.exit(main())
