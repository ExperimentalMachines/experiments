# AWS export boxes, archived 2026-11-09

What was worth keeping from the four EC2 export boxes in account <aws-account> before they were terminated, at the user's request, so that nothing on AWS bills any more.

| Box | Instance | Type | Region | Used for |
|---|---|---|---|---|
| Q1 | <instance> | r7i.4xlarge on-demand | us-east-1a | LFM3 builds, Qwen2.5-1.5B re-export, GPTQ-Vulkan rollout |
| Q3 | <instance> | r7i.8xlarge persistent spot | us-east-1a | Qwen3 32k builds, rollout |
| P1 | <instance> | r7i.4xlarge on-demand | us-west-2b | Phi-4-mini, rollout, Qwen3-0.6B audit |
| P3 | <instance> | r7i.8xlarge persistent spot | us-west-2b | Phi-4-mini 32k, rollout, Qwen3-4B-Thinking check |

## Layout

- `codes/`: every distinct GPTQ solve (`codes.pt`) found on the boxes, zstd-compressed, named `<first 12 hex of sha256>-<run>.codes.pt.zst`. The hash is of the uncompressed file, and each was checked after download (`zstd -dc <file> | shasum -a 256`). `inventory/sha-*.txt` lists every copy on every box, including duplicates that were not downloaded twice. Restore with `zstd -d`.
- `Q1/`, `Q3/`, `P1/`, `P3/`: every file under 50 MB in each box's home directory, at its original path. That is run logs, export and verify reports, `gate.json` fp32 references (compute these on Linux only; the Mac's are wrong, research finding 44), byte-check (`codecheck/`) results, and the run scripts (`run-export.sh`, `run-lfm3.sh`, `job*.sh`, `parity.py`, `expt.py`, `think_check.py`).
- `P1/runs/gptq-smollm2-360m-instruct/out-xnnpack-*/`: the unpublished GPTQ CPU builds of SmolLM2-360M-Instruct, one per window. They are the gate proxies that the published Vulkan files' reports cite by sha256.
- `inventory/`: each box's file list (`files-*.tsv`, size and path), disk survey, codes.pt hashes, and the EC2 instance, volume and spot-request descriptions as they were before termination.

## Not kept, and where to get it back

- Source weights (`.safetensors`, converted `.pth` checkpoints) and Hugging Face caches: re-download from the Hub at the revision in each export report.
- Published `.pte` files: on huggingface.co/experimentalmachines. Earlier versions are in each repo's history.
- Experiment builds that are not published: rebuild from the solve in `codes/` with execupack at the commit in the run's report. These are the Qwen3-0.6B int4 builds (finding 43), the Phi-4-mini 2k and round-to-nearest builds, LFM3's ExecuTorch 1.6 and fp32 builds, and the Qwen3-4B-Thinking 4k check.
- Python venvs and ExecuTorch toolchains: reinstall from execupack's `requirements/`.
- No HF token, SSH key or other credential was copied. The boxes' `.config/` and `.ssh/` were excluded, and their files were scanned for `hf_` tokens first.
