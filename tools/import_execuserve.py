#!/usr/bin/env python3
"""ExecuServe's experiment record: `docs/results/` (one write-up and one evidence folder per
study), the memory-limits study that was never committed, and the logs of the two QNN
exports the NPU study ran.

What stays in ExecuServe on purpose: the app, `tools/execuserve` (the command-line driver the
studies call through `$REPO_ROOT`), `tools/compat` (conformance checks the README and
CONTRIBUTING.md send people to, though bench.py and soak.py only measure), and
`tools/executorch` (it builds the shipped runtime; its cpu_forward_bench.cpp is compiled by
build_mediatek.py).

  python3 tools/import_execuserve.py [--check]
"""

from pathlib import Path

from importer import Edit, Project, Source, main
import serials

HOME = Path.home()
SHA = "bfbfa85353c3d57cb0d61c0ca49ba3ec1e6ca01c"
X = "execuserve/studies"
ONE_SIX = "docs/results/2026-11-13-executorch-1.6.0/phone"
SERIAL_FROM_ENV = "SER=${SER:?set SER to the adb serial of the phone}"

UNITS = serials.replacer(serials.discover(HOME / "execuserve", SHA, ["docs/results"])
                         | serials.discover_files((HOME / "execuserve/docs/results/2026-11-04-memory-limits").rglob("*.*")))

RESULTS = Source(
    label="execuserve@main (bfbfa853)",
    url="https://github.com/ExperimentalMachines/execuserve",
    checkout=HOME / "execuserve",
    commit=SHA,
    roots=["docs/results"],
    worktree=["docs/results/2026-11-04-memory-limits.md", "docs/results/2026-11-04-memory-limits"],
    moves=[("docs/results", X)],
    edits={
        # The phone's serial was written into the scripts; it comes from the environment here.
        f"{ONE_SIX}/run.sh": [Edit(r"SER=[A-Za-z0-9]+", SERIAL_FROM_ENV, regex=True)],
        f"{ONE_SIX}/run2.sh": [Edit(r"SER=[A-Za-z0-9]+", SERIAL_FROM_ENV, regex=True)],
        "docs/results/2026-11-04-memory-limits/memtest.py": [
            Edit(r'S = "[0-9a-f]{8}"', 'S = os.environ["SER"]  # the phone\'s adb serial', regex=True)],
    },
    redact=[
        # A Qualcomm device-cloud session: a support address, the SSH login, and the internal
        # cluster name of the device the tunnel reached.
        ("docs/results/2026-10-04-vulkan/*.log", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]*qualcomm\.com", "<redacted-address>"),
        ("docs/results/2026-10-04-vulkan/*.log", r"\b[a-z]{2}\d{6}\.sa\.svc\.cluster\.local", "<device>.sa.svc.cluster.local"),
    ],
    hooks=[UNITS],
)

EXPORTS = Source(
    label="Desktop/execuserve-play/qnn (the NPU study's QNN exports, 2026-10-06)",
    checkout=HOME / "Desktop/execuserve-play",
    commit=None,
    roots=["qnn"],
    moves=[("qnn", f"{X}/2026-10-06-npu/exports")],
    suffixes={".log", ".json", ".txt", "started"},
    exclude={"tokenizer*.json": "tokenizer, downloadable from the model's repository"},
    max_bytes=6_000_000,
    rewrite="none",
)

PROJECT = Project("execuserve", [RESULTS, EXPORTS])

if __name__ == "__main__":
    main(PROJECT)
