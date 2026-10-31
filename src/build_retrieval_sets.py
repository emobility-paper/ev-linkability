#!/usr/bin/env python3
"""Stage E2a: build the candidate-set retrieval index (section 14).

Reorganizes the split map into a retrieval-ready index: for each identity, its
training sessions (the reference pool the attacker learns from) and its test
sessions (the positives to be found), plus the global pool of all test sessions
(the distractor source). No scoring happens here -- this just defines the
retrieval population, deterministically.

Run (after split_sessions):
    python src/build_retrieval_sets.py --splits data/processed/splits_min20_user_site.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    splits_path = Path(args.splits)
    rows = list(csv.DictReader(splits_path.open(newline="", encoding="utf-8")))

    train_by: dict = defaultdict(list)
    test_by: dict = defaultdict(list)
    test_pool = []
    for r in rows:
        if r["split"] == "train":
            train_by[r["user_hash"]].append(r["sessionID"])
        elif r["split"] == "test":
            test_by[r["user_hash"]].append(r["sessionID"])
            test_pool.append([r["sessionID"], r["user_hash"]])

    targets = {}
    for uh in set(train_by) | set(test_by):
        if train_by[uh] and test_by[uh]:  # need at least one reference and one positive
            targets[uh] = {"train": sorted(train_by[uh]), "test": sorted(test_by[uh])}

    tag = splits_path.stem.replace("splits_", "")
    out = Path(args.out) if args.out else splits_path.parent / f"retrieval_index_{tag}.json"
    out.write_text(json.dumps({
        "tag": tag,
        "n_targets": len(targets),
        "test_pool_size": len(test_pool),
        "targets": targets,
        "test_pool": test_pool,
    }))
    print(f"retrieval index: {len(targets)} eligible targets, "
          f"{len(test_pool)} test-pool sessions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
