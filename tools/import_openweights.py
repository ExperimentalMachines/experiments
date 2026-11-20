#!/usr/bin/env python3
"""OpenWeights' experiment material: the device harness, its results, and the host evals.

What stays in OpenWeights on purpose: the instrumented benchmarks under `app/` and `core/`
(they compile against the engine), the write-ups in `docs/research/` (the KDoc cites them),
and the product tooling under `tools/` (release, hooks, review, the NPU runtime build,
the ExecuTorch export and template references, and the two test-image generators the
instrumented tests use).

  python3 tools/import_openweights.py [--check]
"""

from pathlib import Path

from importer import Edit, Project, Source, main
from logcat_filter import filter_xz_logcat
import serials
from archives import rewrite_tar

OW = "${OPENWEIGHTS:?set OPENWEIGHTS to an openweights checkout}"

RUNS = {
    # prefill and decode on each processor, the thread and throttle runs, the stall guard
    "prefill-decode": [
        "tools/snapdragon/results/2026-10-24-sm8850",
        "tools/snapdragon/results/2026-11-09-poco-mt6991",
        "tools/snapdragon/results/2026-11-10-sm8550",
        "tools/snapdragon/results/2026-11-11-sm8850-r52",
        "tools/snapdragon/results/2026-11-16-s24",
        "tools/mediatek/results/2026-11-16-mt6989",
        "tools/mediatek/results/2026-11-18-mt6993-pilot",
    ],
    "runtime-upgrades": [
        "tools/snapdragon/results/2026-10-31-llama-v080",
        "tools/snapdragon/results/2026-11-04-poco-v080",
        "tools/snapdragon/results/2026-11-04-qdc-v080",
        "tools/snapdragon/results/2026-11-14-runtime-upgrade",
    ],
    "shipped-build": [
        "tools/snapdragon/results/2026-11-05-qdc-last-round",
        "tools/snapdragon/results/2026-11-05-qdc-shipped",
        "tools/snapdragon/results/2026-11-06-emulator-release",
        "tools/snapdragon/results/2026-11-06-qdc-final",
    ],
    "prefix-reuse": [
        "tools/snapdragon/results/2026-11-15-coldwarm-poco",
        "tools/snapdragon/results/2026-11-16-coldwarm-matrix",
        "tools/snapdragon/results/2026-11-16-reuse-boundary",
    ],
    "retrieve-or-answer": [
        "tools/eval/results/decisions",
        "tools/eval/results/gate",
        "tools/eval/results/confidence",
        "tools/eval/results/typesafe",
    ],
    "compiled-tool-calling": ["tools/eval/results/toolcall-probe"],
    "vulkan": ["tools/eval/results/vulkan-2026-10-04"],
    "test-lab-archive": ["tools/eval/results/ftl"],
}

S = "openweights/studies"
H = "openweights/harness"

MOVES = [(src, f"{S}/{study}/runs/{Path(src).name}") for study, srcs in RUNS.items() for src in srcs]
MOVES += [
    # The flat folder stays whole: compare.py, bench/report.py and bench/window_report.py
    # each read it as one pool and pick their own files by content.
    ("tools/eval/results", f"{S}/on-device-evals/runs"),
    ("tools/eval/streaming/npu_access/check_npu.sh", f"{H}/device-streaming/check_npu.sh"),
    ("tools/eval/streaming/npu_access", f"{S}/prefill-decode/runs/npu-access"),
    ("tools/eval/streaming", f"{H}/device-streaming"),
    ("tools/eval", f"{H}/app-evals"),
    ("tools/snapdragon/results", f"{S}/prefill-decode/runs"),
    ("tools/snapdragon", f"{H}/prefill-decode/snapdragon"),
    ("tools/mediatek/results", f"{S}/prefill-decode/runs"),
    ("tools/mediatek", f"{H}/prefill-decode/mediatek"),
    ("tools/memory/results", f"{S}/memory-paging/runs"),
    ("tools/memory", f"{H}/memory"),
    ("tools/npu/multiturn", f"{H}/npu/multiturn"),
    ("tools/npu/npu_matmul_bench.cpp", f"{H}/npu/npu_matmul_bench.cpp"),
    ("tools/npu/cpu_matmul_bench.cpp", f"{H}/npu/cpu_matmul_bench.cpp"),
    ("tools/npu/cpu_forward_bench.cpp", f"{H}/npu/cpu_forward_bench.cpp"),
    ("tools/npu/README.md", f"{H}/npu/README.md"),
    ("tools/vision-probe-image.py", f"{H}/host-prompt/vision-probe-image.py"),
]
HOST = ["behaviour_probe.py", "date_notools_eval.py", "date_placement_eval.py",
        "date_structure_eval.py", "image_speed_probe.py", "prompt_dump.json", "routing_matrix.py"]
MOVES += [(f"eval/{name}", f"{H}/host-prompt/{name}") for name in HOST]

ROOTS = ["tools/eval", "tools/snapdragon", "tools/mediatek", "tools/memory", "tools/npu/multiturn",
         "tools/npu/npu_matmul_bench.cpp", "tools/npu/cpu_matmul_bench.cpp",
         "tools/npu/cpu_forward_bench.cpp", "tools/npu/README.md", "tools/vision-probe-image.py", "eval"]

UNTRACKED = [
    "tools/mediatek/results/2026-11-18-mt6993-pilot",
    "tools/snapdragon/results/2026-10-24-sm8850/chain29.out",
    "tools/snapdragon/results/2026-10-24-sm8850/chain49.out",
    "tools/snapdragon/results/2026-11-04-poco-v080/debug-v080-chat.png",
    "tools/snapdragon/results/2026-11-04-poco-v080/release-smoke-chat.png",
]

EXCLUDE = {
    # A tokenizer is a model file, and compare.py's pool reader crashes on its `model` object.
    "tools/eval/results/elite-LFM2.5-2.6B-8da4w-32k.tokenizer.json": "tokenizer, downloadable with its model",
    "tools/eval/results/elite-LFM2.5-2.6B-8da4w-4k.tokenizer.json": "tokenizer, downloadable with its model",
}


def root_is_openweights(old: str) -> Edit:
    return Edit(old, f"ROOT={OW}")


EVALS_OUT = "$HERE/../../../studies/on-device-evals/runs"
DECISIONS_OUT = "$HERE/../../../studies/retrieve-or-answer/runs/decisions"
BENCH_ROOT = 'ROOT=$(cd "$HERE/../../.." && pwd)'

EDITS = {
    "tools/eval/run_matrix.sh": [
        root_is_openweights('ROOT=$(cd "$HERE/../.." && pwd)'),
        Edit('OUT="$HERE/results"', 'OUT="$HERE/../../studies/on-device-evals/runs"'),
    ],
    "tools/eval/run_matrix_ftl.sh": [
        root_is_openweights('ROOT=$(cd "$HERE/../.." && pwd)'),
        Edit('OUT="$HERE/results"', 'OUT="$HERE/../../studies/on-device-evals/runs"'),
    ],
    "tools/eval/bench/run_cloud.sh": [
        root_is_openweights(BENCH_ROOT),
        Edit('OUT="$HERE/../results"', f'OUT="{EVALS_OUT}"'),
        Edit("""'"$ROOT"'/tools/eval/run_matrix_ftl.sh""", """'"$HERE"'/../run_matrix_ftl.sh"""),
    ],
    "tools/eval/bench/run_cloud_rerun.sh": [
        root_is_openweights(BENCH_ROOT),
        Edit('OUT="$HERE/../results"', f'OUT="{EVALS_OUT}"'),
        Edit('"$ROOT/tools/eval/run_matrix_ftl.sh"', '"$HERE/../run_matrix_ftl.sh"'),
    ],
    "tools/eval/bench/run_cloud_windows.sh": [
        root_is_openweights(BENCH_ROOT),
        Edit('OUT="$HERE/../results"', f'OUT="{EVALS_OUT}"'),
        Edit("""'"$ROOT"'/tools/eval/run_matrix_ftl.sh""", """'"$HERE"'/../run_matrix_ftl.sh"""),
    ],
    "tools/eval/bench/run_decisions.sh": [
        root_is_openweights(BENCH_ROOT),
        Edit('OUT="$HERE/../results/decisions"', f'OUT="{DECISIONS_OUT}"'),
    ],
    "tools/eval/bench/run_decisions_cloud.sh": [
        root_is_openweights(BENCH_ROOT),
        Edit('OUT="$HERE/../results/decisions"', f'OUT="{DECISIONS_OUT}"'),
    ],
    "tools/eval/bench/run_decisions_ftl.sh": [
        root_is_openweights(BENCH_ROOT),
        Edit('OUT="$HERE/../results/decisions"', f'OUT="{DECISIONS_OUT}"'),
    ],
    "tools/eval/bench/run_decisions_ftl_local.sh": [
        Edit(f'{BENCH_ROOT}; OUT="$HERE/../results/decisions"', f'ROOT={OW}; OUT="{DECISIONS_OUT}"'),
        Edit("$ROOT/eval/prompt_dump.json", "$HERE/../../host-prompt/prompt_dump.json"),
    ],
    "tools/eval/bench/run_ftl_local.sh": [
        Edit(f'{BENCH_ROOT}; OUT="$HERE/../results"', f'ROOT={OW}; OUT="{EVALS_OUT}"'),
    ],
    "tools/eval/bench/run_local.sh": [
        root_is_openweights(BENCH_ROOT),
        Edit('OUT="$HERE/../results"', f'OUT="{EVALS_OUT}"'),
    ],
    # The write-ups these render stay in OpenWeights.
    "tools/eval/bench/report.py": [Edit("--out docs/research/", '--out "$OPENWEIGHTS"/docs/research/')],
    "tools/eval/bench/window_report.py": [Edit("--out docs/research/", '--out "$OPENWEIGHTS"/docs/research/')],
    "tools/eval/bench/run_rtl.sh": [
        Edit('OUT="$HERE/../results"', f'OUT="{EVALS_OUT}"'),
        Edit("APK=$HERE/../../../core/engine/", f"APK={OW}/core/engine/"),
    ],
    "tools/eval/bench/sweep.sh": [Edit('OUT="$HERE/../results"', f'OUT="{EVALS_OUT}"')],
    "tools/eval/bench/pull_matrix.sh": [Edit('OUT="$HERE/../results"', f'OUT="{EVALS_OUT}"')],
    # bench/ sits at openweights/harness/app-evals/bench, so parents[2] of it is openweights/.
    "tools/eval/bench/gate.py": [
        Edit('OUT = HERE.parent / "results" / "gate"',
             'OUT = HERE.parents[2] / "studies" / "retrieve-or-answer" / "runs" / "gate"'),
    ],
    "tools/eval/bench/confidence.py": [
        Edit('OUT = HERE.parent / "results" / "confidence"',
             'OUT = HERE.parents[2] / "studies" / "retrieve-or-answer" / "runs" / "confidence"'),
        Edit('HEADER_FROM = HERE.parent / "results" / "decisions"',
             'HEADER_FROM = HERE.parents[2] / "studies" / "retrieve-or-answer" / "runs" / "decisions"'),
    ],
    "tools/eval/bench/local_sufficiency.py": [
        Edit('Path(__file__).resolve().parent.parent / "results" / "typesafe"',
             'Path(__file__).resolve().parents[3] / "studies" / "retrieve-or-answer" / "runs" / "typesafe"'),
    ],
    "tools/eval/bench/typesafe_matchers.py": [
        Edit('Path(__file__).resolve().parent.parent / "results" / "decisions"',
             'Path(__file__).resolve().parents[3] / "studies" / "retrieve-or-answer" / "runs" / "decisions"',
             count=2),
    ],
    "tools/eval/bench/typesafe_results.py": [
        Edit('Path(__file__).resolve().parent.parent / "results" / "decisions"',
             'Path(__file__).resolve().parents[3] / "studies" / "retrieve-or-answer" / "runs" / "decisions"'),
    ],
    "tools/eval/bench/toolcall_probe/prompts.py": [
        Edit('"..","..","results","decisions"',
             '"..","..","..","..","studies","retrieve-or-answer","runs","decisions"'),
    ],
    "tools/eval/streaming/README.md": [
        Edit("app integration captures stay under `../results/`",
             "app integration captures go under `../../studies/<study>/runs/`"),
    ],
    "tools/eval/streaming/npu_access/check_npu.sh": [
        Edit('ST=$(cd "$HERE/.." && pwd); OUT=$HERE/$LAB',
             'ST=$HERE; OUT=$HERE/../../studies/prefill-decode/runs/npu-access/$LAB'),
    ],
    "tools/mediatek/build_npu_bench_linux.sh": [
        Edit('REPO="$(cd "$HERE/../.." && pwd)"', f"REPO={OW}"),
    ],
    "tools/mediatek/session.sh": [
        Edit('HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)',
             'HERE=$(cd "$(dirname "$0")" && pwd)'),
        # The archived folder is a record; a new session names its own (2026-11-16-mt6989 ran here).
        Edit("RES=$HERE/results/2026-11-16-mt6989",
             "RES=${RES:?set RES to a new run folder under studies/prefill-decode/runs}"),
        Edit("ST=$ROOT/tools/eval/streaming", "ST=$HERE/../../device-streaming"),
    ],
    "tools/snapdragon/session_sd.sh": [
        Edit('HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)',
             'HERE=$(cd "$(dirname "$0")" && pwd)'),
        # The archived folder is a record; a new session names its own (2026-11-16-s24 ran here).
        Edit("RES=$HERE/results/2026-11-16-s24",
             "RES=${RES:?set RES to a new run folder under studies/prefill-decode/runs}"),
        Edit("ST=$ROOT/tools/eval/streaming", "ST=$HERE/../../device-streaming"),
    ],
    "tools/snapdragon/llama_bench_ab.sh": [
        root_is_openweights('ROOT=$(cd "$(dirname "$0")/../.." && pwd)'),
    ],
    # The export VM's project came from the environment's default; it has to be named now.
    "tools/mediatek/fetch_export.sh": [
        Edit(r"\$\{MTK_PROJECT:-[a-z0-9-]+\}", "${MTK_PROJECT:?set MTK_PROJECT to the project of the export VM}", regex=True),
    ],
    "tools/memory/run_prefetch_matrix.sh": [
        root_is_openweights('ROOT=$(cd "$HERE/../.." && pwd)'),
        Edit("OUT=${OUT:-$HERE/results/matrix-$RUN}",
             "OUT=${OUT:-$HERE/../../studies/memory-paging/runs/matrix-$RUN}"),
    ],
    "tools/memory/run_pressure.sh": [
        root_is_openweights('ROOT=$(cd "$HERE/../.." && pwd)'),
        Edit("OUT=${OUT:-$HERE/results/$(date +%Y-%m-%d)}",
             "OUT=${OUT:-$HERE/../../studies/memory-paging/runs/$(date +%Y-%m-%d)}"),
    ],
}

VULKAN = "tools/eval/results/vulkan-2026-10-04/*"

# Every device serial a run record names becomes a stable pseudonym (tools/serials.py).
COMMIT = "81734deed5048f5691518d0e739662c50e51e1a3"
CHECKOUT = Path.home() / "mobile-inference"
UNITS = serials.replacer(
    serials.discover(CHECKOUT, COMMIT, ["tools", "eval"])
    | serials.discover_files(f for item in UNTRACKED for f in (
        (CHECKOUT / item).rglob("*") if (CHECKOUT / item).is_dir() else [CHECKOUT / item]) if f.is_file()))
PROJECTS = serials.project_replacer(serials.discover_projects(CHECKOUT, COMMIT, ["tools", "eval"]))
ACCOUNTS = serials.account_replacer(serials.discover_accounts(CHECKOUT, COMMIT, ["tools", "eval"]))

PROJECT = Project("openweights", [Source(
    label="openweights@81734dee",
    url="https://github.com/ExperimentalMachines/openweights",
    checkout=Path.home() / "mobile-inference",
    commit="81734deed5048f5691518d0e739662c50e51e1a3",
    roots=ROOTS,
    moves=MOVES,
    worktree=UNTRACKED,
    exclude=EXCLUDE,
    edits=EDITS,
    # The cloud labs' logcats are whole devices' logs; only the app's lines come across. The
    # full logs stay in OpenWeights at this commit.
    transforms=[("*logcat.xz", lambda path, data: filter_xz_logcat(path, data, UNITS, PROJECTS, ACCOUNTS)),
                ("*.tar", lambda path, data: rewrite_tar(path, data, UNITS, PROJECTS, ACCOUNTS))],
    hooks=[UNITS, PROJECTS, ACCOUNTS],
    # The phone's wireless-debugging name carries its serial.
    redact=[
        ("*", r"adb-[A-Za-z0-9]{12,20}-[A-Za-z0-9]{6}\._adb-tls-connect", "adb-<serial>-<id>._adb-tls-connect"),
        # The Vulkan session's logs: Qualcomm device-cloud addresses and cluster names, and
        # the phone's address on the home network.
        (VULKAN, r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]*qualcomm\.com", "<redacted-address>"),
        (VULKAN, r"\b[a-z]{2}\d{6}\.sa\.svc\.cluster\.local", "<device>.sa.svc.cluster.local"),
        (VULKAN, r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "<lan-ip>"),
    ],
)])

if __name__ == "__main__":
    main(PROJECT)
