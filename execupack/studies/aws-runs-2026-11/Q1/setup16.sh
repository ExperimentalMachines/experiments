set -euxo pipefail
export PATH=$HOME/.local/bin:$PATH
cd ~
. et16/bin/activate
uv pip install "executorch @ https://download.pytorch.org/whl/test/cpu/executorch-1.6.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl" --extra-index-url https://download.pytorch.org/whl/test/cpu --index-strategy unsafe-best-match
uv pip install transformers==5.22.1 huggingface_hub==1.31.0 safetensors==0.8.0 jinja2==3.1.6 pyyaml==6.0.3 pytorch-tokenizers==1.5.0 || true
python -c "import torch, executorch, torchao, transformers; from executorch import version; print(torch.__version__, version.__version__, torchao.__version__, transformers.__version__)"
echo SETUP16-DONE
