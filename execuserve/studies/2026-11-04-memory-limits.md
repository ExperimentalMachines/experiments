# Android 17's memory limits, tried by hand, 2026-11-04

Android 17 caps how much memory an app may use by whether it is on screen, and a foreground
service counts as not visible. This page measures what ExecuServe's CPU builds really use and
what happens to them at each phone size's limit. The files it rests on are in
[2026-11-04-memory-limits/](2026-11-04-memory-limits/).

**This is not Android 17.** The phone runs Android 16, so the limiter itself never ran. The same
two kernel settings the limiter uses were applied by hand. That shows the slowdown; it cannot show
whether Android 17 would end the app.

## How it was tested

- QDC phone: SM8850, Android 16, 23,203 MiB of memory, 6 GiB of swap, on AC, screen on, another
  app (idle) in front. ExecuServe 0.1.1 (build 73), release.
- ExecuServe was in `curProcState=4` (foreground service) throughout, which is the limiter's "not
  visible" row.
- Per-app memory accounting was turned on with `+memory` in `/sys/fs/cgroup/apps/cgroup.subtree_control`,
  and turned off again afterwards.
- For each run: stop the app, drop the file cache, start the process, set `memory.high` and
  `memory.swap.max` to the [limiter's not-visible values](https://source.android.com/docs/core/perf/memory-limiter)
  on the app's uid cgroup, load the model, then send two short requests (about 25 tokens in, 64 out)
  and one long one (about 3,500 of the 4,096-token window, which forces the conversation cache into
  use). Greedy; one process for the whole run; 180 s timeout per request.

## What the models really use

| Model, 4k window | File | App's estimate (`ModelMemory`) | Measured anonymous memory | Measured file-backed mappings |
|---|---:|---:|---:|---:|
| LFM2.5-1.2B | 0.80 GB | 1.13 GiB | 1,052 MiB | 15 MiB |
| Qwen3-1.7B | 1.29 GB | 2.37 GiB | 2,248 MiB | 15 MiB |
| Qwen3-4B | 2.66 GB | 3.90 GiB | 3,821 MiB | 15 MiB |

Nearly all of it is ordinary app memory, which can only go to swap. ExecuTorch's Android default
memory-maps the file, but the weights end up in anonymous memory anyway. Reading the file also
leaves page cache charged to the app (about the file's size), which the limit simply drops.
`ModelMemory`'s estimate is within about 10% of the measurement.

## Results

Times are the server's own (`/v1/execuserve/runs`). "Short" is decode tokens per second over the two
short requests; "long" is the whole long request.

| Model | Limit (phone size) | Short, tok/s | Long, seconds | Swap at the end of the long request | Swap allowance full? |
|---|---|---:|---:|---:|---|
| LFM2.5-1.2B | none | 84-88 | 7.6 | 0 | |
| | 2 GiB (6 GB) | 84-88 | 7.7 | 0 | no |
| | 1 GiB (4 GB) | 82-87 | 7.8 | 399 MiB | no |
| Qwen3-1.7B | none | 55-57 | 21.3 | 1 MiB | |
| | 3 GiB (8 GB) | 54-57 | 21.5 | 0 | no |
| | 2 GiB (6 GB) | 53-57 | 22.2 | 230 MiB | no |
| | 1 GiB (4 GB) | 44-55 | **not finished in 180 s** | 1,023 MiB | **yes** |
| Qwen3-4B | none | 25-26 | 47.4 | 0 | |
| | 5 GiB (16 GB) | 25 | 49.0 | 320 MiB | no |
| | 4 GiB (12 GB) | 24-25 | 49.9 | 300 MiB | no |
| | 3 GiB (8 GB) | 24-25 | **82.6** (decode 0.77 tok/s, from 4.93) | 1,535 MiB | no |
| | 2 GiB (6 GB) | **1.0-1.2** | **173.7** | 2,046 MiB after loading, 1,934 MiB here | **yes** |

- **The process was never killed.** All 12 runs ended in the process they started in, with no
  out-of-memory events. Android 16 has no limiter to end it.
- **A short chat costs less than the model's size suggests.** The conversation cache is mostly empty
  and goes to swap for free, which is why Qwen3-1.7B still answered short questions under 1 GiB.
  A long conversation touches all of it.
- **The two runs with the swap allowance full** are the situation in which Android 17's documentation
  says an app can be terminated. Whether it would be, these runs cannot say. On Android 16 the kernel
  only slowed it down: `memory.swap.events` `fail` rose by about 19 million and 14 million.

## What the app does today

`refuseIfTooBig` refuses a build that needs more than 65% of the phone's memory. Android 17's limits
for a foreground service are 25 to 38% of a phone's nominal memory, and apps cannot ask for them at
run time. So, by the app's own arithmetic against the limiter's published table:

| Build | Phone | Need vs limit | App's answer | Measured here |
|---|---|---:|---|---|
| Qwen3-1.7B 4k | 4 GB, if `MemTotal` is at least 3,731 MiB | 2.37× | allowed | long chats stall |
| Qwen3-4B 4k | 6 GB, if `MemTotal` is at least 6,140 MiB | 1.95× | allowed | about 20 times slower throughout |
| Qwen3-4B 4k | 8 GB | 1.30× | allowed | long chats about 1.7 times slower |
| Qwen3-1.7B 8k | 6 GB, if `MemTotal` is at least 5,110 MiB | 1.62× | allowed | not tested |
| Qwen3-1.7B 16k | 8 GB, if `MemTotal` is at least 7,867 MiB | 1.66× | allowed | not tested |
| Qwen3-1.7B 32k | 16 GB | 1.70× | allowed | not tested |

## What this does not show

- Whether Android 17 kills the app. That needs Android 17: `am memory-limiter manual <pid> <MB>` is the
  documented way to force a limit, and the kill would read `REASON_OTHER` with `MemoryLimiter:AnonSwap`.
- How a real 4 to 8 GB phone behaves. This one has a flagship CPU and fast swap, and no memory pressure
  from anything else, so a real one should do worse once it starts swapping.
- Windows above 4k, several models at once (the default is one), and the GPU and NPU, whose memory
  was not part of this.
- Whether the stalled Qwen3-1.7B request would have finished; the client gave up at 180 s and the server
  recorded `client_gone`.
- A possible restart loop: a cancelled native call that takes more than `wedgeGraceMs` (30 s) ends the
  process, and the restart could stall the same way. It did not fire in these runs, but each run ended
  soon after its last request.
- The limit a client on screen would give ExecuServe through a bound service (5 GiB on an 8 GB phone,
  per the limiter's table). Qwen3-4B was fine at 5 GiB here, but no bound client was tried.
