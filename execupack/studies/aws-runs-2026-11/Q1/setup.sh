set -euxo pipefail
cd ~/execupack
export PATH=$HOME/.local/bin:$PATH
uv venv --python 3.11 .venv
. .venv/bin/activate
uv pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements/export-xnnpack.txt
ls requirements/
uv pip install -r requirements/dev.txt || true
python -c "import torch, executorch, transformers; print(torch.__version__, transformers.__version__)"
echo SETUP-DONE
