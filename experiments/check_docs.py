"""check-docs — verify (and optionally update) the result numbers quoted in the
markdown docs against the pipeline's results manifests.

Every quoted figure is wrapped in an inline directive, invisible in rendered
markdown:

    <!--n <manifest>: <template>-->current text<!--/n-->

`<manifest>` names `<root>/results/<manifest>.json` (written by the experiment
via `dota2ad.eval.results.write_results`); `<template>` is a str.format string
over that manifest's fields. The wrapped span must equal the rendered template
(unicode minus / NBSP normalized). The directive lives next to the text it
checks, so coverage is visible while editing and quoting a figure twice means
wrapping it twice.

Modes:
    check (default) — exit 1 on any STALE / MISSING / NO-FIELD directive.
    --fix           — rewrite stale spans in place from the manifests and list
                      every change. Only the digits move; the surrounding prose
                      (significance marks, "clears zero" wording) is yours —
                      review each listed site, the interpretation may need to
                      move with the numbers.
    --audit         — informational: numeric tokens OUTSIDE any directive, per
                      doc, so newly added numbers that should be wrapped are
                      visible. No effect on the exit code (structural
                      constants, worked examples, and citations are
                      legitimately unwrapped).

Deliberately unchecked number classes (do not wrap): structural game constants
(48/36/12/10/50 items, timer seconds), worked-example arithmetic (the 12·β̂
translations), rounded restatements in flowing prose, per-seed slash-lists,
cross-manifest derived figures, citation years, and the patch/collection-window
identifiers.

Run:  DOTA2AD_ROOT=work pixi run check-docs [--fix] [--audit]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from dota2ad.core.paths import default_paths

REPO = Path(__file__).resolve().parent.parent
DIRECTIVE = re.compile(r"<!--n ([\w-]+): (.*?)-->(.*?)<!--/n-->", re.S)
NUM_TOKEN = re.compile(r"[+−-]?\d[\d,]*(?:\.\d+)?%?")


def _norm(s: str) -> str:
    return s.replace("−", "-").replace(" ", " ")


def _tracked_docs() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "*.md"], capture_output=True,
                         text=True, check=True, cwd=REPO)
    return [REPO / p for p in out.stdout.split() if not p.startswith("_archive/")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix", action="store_true",
                    help="rewrite stale spans from the manifests")
    ap.add_argument("--audit", action="store_true",
                    help="list numeric tokens outside directives (informational)")
    args = ap.parse_args()

    results_dir = default_paths().root / "results"
    manifests: dict[str, dict | None] = {}

    def manifest(name: str) -> dict | None:
        if name not in manifests:
            p = results_dir / f"{name}.json"
            manifests[name] = json.loads(p.read_text()) if p.exists() else None
        return manifests[name]

    n_ok = n_stale = n_missing = n_nofield = n_fixed = 0
    for doc in _tracked_docs():
        text = doc.read_text()
        rel = doc.relative_to(REPO)
        if args.audit:
            outside = DIRECTIVE.sub("", text)
            toks = NUM_TOKEN.findall(outside)
            if toks:
                print(f"audit    {rel}: {len(toks)} numeric tokens outside directives")
            continue

        changed = False
        out, pos = [], 0
        for m in DIRECTIVE.finditer(text):
            name, template, span = m.group(1), m.group(2), m.group(3)
            template = template.replace("\\|", "|")  # pipes are escaped in-comment for GFM tables
            data = manifest(name)
            label = f"{rel}: {template.strip()[:70]}"
            if data is None:
                print(f"MISSING  {name}.json — run the pipeline ({label})")
                n_missing += 1
                continue
            try:
                rendered = template.format(**data)
            except KeyError as e:
                print(f"NO-FIELD {name}.json lacks {e} ({label}) — the run suppressed "
                      f"this figure; the claim needs rewording, not just numbers")
                n_nofield += 1
                continue
            if _norm(rendered) == _norm(span):
                n_ok += 1
            elif args.fix:
                out.append(text[pos:m.start(3)])
                out.append(rendered)
                pos = m.end(3)
                changed = True
                n_fixed += 1
                print(f"FIXED    {rel}:\n         was  {span}\n         now  {rendered}\n"
                      f"         review the surrounding prose — interpretation may move too")
            else:
                print(f"STALE    {rel}:\n         doc      {span}\n         manifest {rendered}")
                n_stale += 1
        if changed:
            out.append(text[pos:])
            doc.write_text("".join(out))

    if args.audit:
        return 0
    print(f"\n{n_ok} ok, {n_stale} stale, {n_fixed} fixed, "
          f"{n_nofield} suppressed, {n_missing} missing manifests")
    return 1 if (n_stale or n_missing or n_nofield) else 0


if __name__ == "__main__":
    raise SystemExit(main())
