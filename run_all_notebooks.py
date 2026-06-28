#!/usr/bin/env python3
"""Execute all tutorial notebooks in-place under the ~/opt/triqs environment.

Reproduces the `unstable_executed` branch: walks every .ipynb (excluding
.ipynb_checkpoints), executes it with nbclient (allow_errors=True), and
saves the result back. IPython `?Foo` / `Foo?` help queries are not visible
to nbconvert's payload mechanism, so before executing we substitute them
with `pydoc.render_doc(...)` calls (the original cell source is restored
afterwards so the notebook still *shows* `?Foo`).

Usage:
    # Make sure the env is set up first
    source ~/opt/triqs/share/triqs/triqsvars.sh
    python3 run_all_notebooks.py                # all notebooks
    python3 run_all_notebooks.py --only Basics  # only matching paths
    NB_TIMEOUT=7200 python3 run_all_notebooks.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {".git", ".ipynb_checkpoints"}

# Pattern for a line that is purely an IPython help query.
# Captures optional leading `?`/`??`, the dotted target, and optional trailing `?`/`??`.
HELP_QUERY = re.compile(
    r"^(?P<lead>\?\??)?(?P<target>[A-Za-z_][\w\.]*)(?P<trail>\?\??)?\s*$"
)


def transform_help_lines(source: str) -> tuple[str, bool]:
    """Replace `?Foo` / `Foo?` lines with pydoc render calls."""
    changed = False
    out: list[str] = []
    for raw in source.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "?" not in stripped:
            out.append(raw)
            continue
        m = HELP_QUERY.match(stripped)
        if not m:
            out.append(raw)
            continue
        lead, trail = m.group("lead") or "", m.group("trail") or ""
        if not (lead or trail):
            out.append(raw)
            continue
        target = m.group("target")
        indent = raw[: len(raw) - len(raw.lstrip())]
        wants_source = lead == "??" or trail == "??"
        out.append(f"{indent}import pydoc as _ihelp_pd")
        out.append(
            f"{indent}print(_ihelp_pd.render_doc({target}, renderer=_ihelp_pd.plaintext))"
        )
        if wants_source:
            out.append(f"{indent}import inspect as _ihelp_in")
            out.append(
                f"{indent}print('\\n--- source ---\\n', _ihelp_in.getsource({target}))"
            )
        changed = True
    return "\n".join(out), changed


def iter_notebooks(root: Path):
    for p in sorted(root.rglob("*.ipynb")):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p


def coalesce_streams(cell) -> None:
    """Merge consecutive stream outputs of the same name (avoids per-line noise)."""
    outputs = cell.get("outputs")
    if not outputs:
        return
    merged: list = []
    for out in outputs:
        if (
            out.get("output_type") == "stream"
            and merged
            and merged[-1].get("output_type") == "stream"
            and merged[-1].get("name") == out.get("name")
        ):
            merged[-1]["text"] = "".join([merged[-1].get("text", ""), out.get("text", "")])
        else:
            merged.append(out)
    cell["outputs"] = merged


def strip_volatile_metadata(nb) -> None:
    md = nb.get("metadata", {})
    md.pop("execution", None)
    li = md.get("language_info")
    if li and "version" in li:
        # Drop the patch-level python version that bumps between hosts.
        del li["version"]
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.metadata.pop("execution", None)


def execute_notebook(path: Path, timeout: int) -> None:
    nb = nbformat.read(path, as_version=4)

    originals: dict[int, str] = {}
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        new_src, changed = transform_help_lines(cell.source)
        if changed:
            originals[i] = cell.source
            cell.source = new_src

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": str(path.parent)}},
    )
    exc: BaseException | None = None
    try:
        client.execute()
    except BaseException as e:  # noqa: BLE001 — capture so we can still persist partial output
        exc = e

    for i, src in originals.items():
        nb.cells[i].source = src
    for cell in nb.cells:
        if cell.cell_type == "code":
            coalesce_streams(cell)
    strip_volatile_metadata(nb)
    nbformat.write(nb, path)

    if exc is not None:
        raise exc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only", default=None,
        help="substring filter on relative notebook path (case-insensitive)",
    )
    ap.add_argument(
        "--timeout", type=int,
        default=int(os.environ.get("NB_TIMEOUT", "3600")),
        help="per-cell timeout in seconds (default: 3600)",
    )
    args = ap.parse_args()

    notebooks = list(iter_notebooks(ROOT))
    if args.only:
        needle = args.only.lower()
        notebooks = [n for n in notebooks if needle in str(n.relative_to(ROOT)).lower()]
    print(f"Executing {len(notebooks)} notebook(s); per-cell timeout={args.timeout}s",
          flush=True)

    results: list[tuple[str, Path, float, str | None]] = []
    for nb in notebooks:
        rel = nb.relative_to(ROOT)
        print(f"\n=== {rel} ===", flush=True)
        t0 = time.time()
        try:
            execute_notebook(nb, timeout=args.timeout)
            elapsed = time.time() - t0
            print(f"OK   {rel} ({elapsed:.1f}s)", flush=True)
            results.append(("OK", rel, elapsed, None))
        except Exception as exc:  # noqa: BLE001 — we want to keep going
            elapsed = time.time() - t0
            tb = traceback.format_exception_only(type(exc), exc)[-1].strip()
            print(f"FAIL {rel} ({elapsed:.1f}s): {tb}", flush=True)
            results.append(("FAIL", rel, elapsed, tb))

    print("\n========== SUMMARY ==========", flush=True)
    for status, rel, elapsed, err in results:
        line = f"{status:>4}  {elapsed:7.1f}s  {rel}"
        if err:
            line += f" — {err}"
        print(line)
    n_ok = sum(1 for r in results if r[0] == "OK")
    n_fail = sum(1 for r in results if r[0] == "FAIL")
    print(f"\n{n_ok} OK, {n_fail} FAIL, {len(results)} total")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
