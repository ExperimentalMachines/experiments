# Glossary

The words the studies use, in plain terms. Each entry says what the word means here, which is
sometimes narrower than its general meaning.

**Jump to:** [How a model answers](#how-a-model-answers) ·
[Model files and runtimes](#model-files-and-runtimes) · [Quantization](#quantization) ·
[Measuring quality](#measuring-quality) · [Phones and the cloud labs](#phones-and-the-cloud-labs) ·
[Memory and heat](#memory-and-heat)

## How a model answers

### Prefill and decode

A reply happens in two phases. **Prefill** reads the whole prompt in one pass, many tokens at
once; it is limited by raw compute. **Decode** then writes the reply one token at a time; it is
limited by how fast the model's weights can be read from memory. Different processors can win
each phase, which is why the [prefill-decode](../openweights/studies/prefill-decode/) study tries
every pairing.

### Time to first token

How long the user waits before the first word of the reply appears: mostly prefill. The
studies report it in milliseconds, often as TTFT.

### Tokens per second

Speed in tokens (word pieces, roughly three quarters of an English word each) per second,
written tok/s. Prefill speed and decode speed are reported separately because they differ by an
order of magnitude.

### KV cache

The model's working memory of the conversation so far: for every token already read, the keys
and values each attention layer computed. Keeping it between turns means the next turn only reads
what is new. Throwing it away means re-reading the whole conversation. See
[prefix reuse](../openweights/studies/prefix-reuse/).

### Warm prefix and snapshot

Reading the fixed start of the prompt (instructions and tool definitions, identical every turn)
before the user asks anything, and saving that state so a new chat starts from it. This is how
OpenWeights cut a first turn from 9.3 s to 0.24 s.

### Hybrid model

A model that mixes attention layers with recurrent or convolution layers (LFM2.5, Qwen3.5). Its
recurrent state cannot be partly rolled back, so a cache can only be reused whole or restored
from a saved checkpoint.

### Context window

The most tokens a model can hold at once. A compiled `.pte` fixes it at export time (2k, 4k up to
32k); a bigger window costs memory from the moment the model loads.

### BOS token

The beginning-of-sequence token some models expect first. Several compiled exports were fed none,
which quietly changed their answers until it was found.

### Greedy decoding

Always picking the single most likely next token (temperature 0). It makes a run repeatable on
one phone, though the studies found it is not identical across phones.

### Tool call

The model asking the app to do something, such as a web search, by writing a structured call
instead of an answer. Several studies measure whether a model still calls tools after it has been
compressed for the phone.

## Model files and runtimes

### GGUF

A model file format read by llama.cpp. It carries its own chat template and metadata.

### .pte

A compiled ExecuTorch program: the model's graph and weights, built for one backend (CPU, GPU or
one vendor's NPU) and one context window. Unlike a GGUF it carries no chat template or stop list,
so the app has to supply them.

### llama.cpp

An open-source C/C++ inference library that runs GGUF files on the CPU, and on some GPUs.
OpenWeights ships it as one of its two runtimes.

### ExecuTorch

PyTorch's on-device runtime, which runs `.pte` files. OpenWeights ships it as its second runtime;
ExecuPack produces the `.pte` files; ExecuServe serves them.

### XNNPACK

ExecuTorch's CPU backend. A `.pte` built for XNNPACK runs on any modern Android phone.

### KleidiAI

Arm's optimized CPU kernels, used by both XNNPACK and llama.cpp on recent Arm cores. Comparing an
accelerator against a CPU that is not using them makes the accelerator look several times better
than it is.

### Vulkan

The cross-vendor graphics and compute API that ExecuTorch's GPU backend uses.

### NPU

The neural processing unit: a phone chip's dedicated accelerator for neural networks (Qualcomm
Hexagon, MediaTek APU). Fast at reading prompts, often slower at writing replies.

### QNN

Qualcomm's AI Engine Direct, the toolchain and runtime for Qualcomm's NPU, and the ExecuTorch
backend of the same name.

### NeuroPilot

MediaTek's toolchain and runtime for its NPU, and the ExecuTorch backend that uses it.

## Quantization

### Quantization

Storing a model's weights (and sometimes its activations) in fewer bits than the 16 or 32 it was
trained with, so it fits in a phone's memory and runs faster. The cost is some drift from the
original model, which can be invisible on benchmarks and visible in behaviour such as tool use.

### 8da4w

ExecuTorch's common phone recipe: 8-bit dynamic activations, 4-bit weights in groups of 32.

### GGUF quantization types

`Q4_K_M`, `Q4_0` and `Q8_0` are GGUF formats: roughly 4-bit with mixed precision, plain 4-bit,
and 8-bit weights. `QAD` marks a Q4_0 file made from quantization-aware weights.

### Round to nearest (RTN)

The simplest way to quantize: round each weight to the nearest 4-bit value. Fast, and the source
of several of the quality losses these studies found.

### GPTQ

A method that chooses the 4-bit values layer by layer to keep each layer's output close to the
original, using a small set of example prompts (the calibration set). ExecuPack's default is to
solve every 4-bit file this way; each export report records whether it was.

### AWQ

Activation-aware weight quantization: scales weights by how strongly each input channel is used
before rounding. Compared with GPTQ in the [AWQ study](../execupack/studies/awq/).

### A16W8 and A16W4

MediaTek NPU recipes: 16-bit activations with 8-bit or 4-bit weights. On MediaTek's NPU the 4-bit
recipe uses one scale per row of weights, much coarser than the CPU recipe.

## Measuring quality

### KL divergence

How far a quantized model's predicted next-token probabilities are from the original model's,
averaged over a text. Zero means identical; ExecuPack uses it to rank recipes.

### Decision gate

ExecuPack's release check: on a fixed set of prompts, does the quantized file make the same
choice as the full-precision model (call a tool, or answer)? A file that fails is not published
unless an operator records an override; Phi-4-mini shipped that way.

### Smoke test

A check that a file loads and generates text. It cannot tell a solved file from a badly rounded
one, which is why the decision gate exists.

### AUROC

How well a score separates two groups (right and wrong answers), from 0 to 1: 0.5 is chance, 1.0
is perfect, and below 0.5 the score points the wrong way. Used for the token-probability signal that decides when to search.

## Phones and the cloud labs

### Firebase Test Lab

Google's farm of real phones in racks. A test APK is uploaded, runs unattended, and its results
come back as files. Most cross-phone runs here used it.

### Firebase Device Streaming

Google's service for borrowing a real phone interactively over adb for a short reservation.

### Qualcomm Device Cloud (QDC)

Qualcomm's lab of reference phones, reachable over an SSH tunnel with a root shell. The Snapdragon
8 Elite Gen 5 runs used it.

### Samsung Remote Test Lab (RTL)

Samsung's lab of Galaxy devices, booked in short sessions. The Dimensity 9500 pilot used it.

### Unit pseudonym

Run records name the physical phone they ran on, because results are compared per unit. Here each
serial number is replaced by `unit-` and six hex characters derived from it, so a unit keeps one
name across every file without its serial being published. Cloud projects and accounts are
replaced the same way.

### Doze

Android's deep idle mode, which defers background work when the screen is off. Benchmarks must
keep the screen on: one dozing phone read prompts at 62 to 307 tok/s against 370 to 378 awake.

## Memory and heat

### Thermal throttling

A hot phone slowing itself down. On the Snapdragon 8 Elite Gen 5 the hard throttle paused six of
eight CPU cores, which made extra threads catastrophically slow; see
[prefill-decode](../openweights/studies/prefill-decode/).

### Swap and page faults

When memory runs short, Android compresses idle memory into swap (zram). A page fault is a read of
memory that has to be brought back first. Being swapped out costs nothing; faulting the weights
back in during a turn is what slows a reply.

### PSS

Proportional set size: the memory a process uses, sharing counted fairly. Android and phone
makers kill apps above limits measured in PSS.
