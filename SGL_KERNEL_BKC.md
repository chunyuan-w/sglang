# BKC: Running xfold (AF3) with sgl-kernel CPU kernels

This is the BKC for enabling the **sgl-kernel** CPU
kernels on top of an existing AlphaFold3 (AF3) Docker image.

The flow is: enter the AF3 container → build and install `sgl-kernel` → check
out the `xfold` branch with the sgl integration → cache the data pipeline once
→ run inference with the sgl kernels selected via env vars.

All paths below are the container paths used in our setup; adjust to your mounts.

---

## 0. Prerequisites

- An existing AF3 Docker image / container with the model params and databases
  mounted (e.g. `/data`, `/data/params`).
- The raw fold inputs available under `/pt211/workspace/inputs`
  (e.g. `amp_81.txt`, `piezo2_2752.txt`, `reelin_3469.txt`, `lrp2_4655.txt`).

---

## 1. Build and install `sgl-kernel` (CPU)

```sh
cd /workspace
git clone https://github.com/chunyuan-w/sglang.git
cd sglang
git checkout chunyuan/fa_with_bias

cd sgl-kernel
cp pyproject_cpu.toml pyproject.toml

# Build toolchain
pip install uv scikit_build_core cmake
python -m pip install intel-openmp==2025.3.3

# tcmalloc allocator (preloaded at runtime; see run_pt211.sh)
apt-get install -y google-perftools
# or, with conda:
# conda install -c conda-forge gperftools=2.18.1

# Use the system GCC/G++ toolchain for the build
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

uv build --wheel -Cbuild-dir=build . --color=always --no-build-isolation
pip3 install "$(find dist -name 'sgl_kernel_cpu-*.whl' | sort | tail -n1)" --force-reinstall
```

This `sglang` repo also ships **`run_pt211.sh`** and **`env.sh`** at its root —
these drive the inference run (steps 3–4). `run_pt211.sh` sources the `env.sh`
sitting next to it; review `env.sh` and adjust only if your container mounts
differ from the defaults:

```sh
cd /workspace/sglang
# edit env.sh if needed: MODEL_DIR, OUTPUT_DIR, NCORES, etc.
```

---

## 2. Check out `xfold` with the sgl integration

`run_pt211.sh` runs xfold's `run_alphafold.py` (it `cd`s into `/pt211/xfold`
internally), so xfold must be checked out at exactly that path:

```sh
mkdir -p /pt211
cd /pt211
git clone https://github.com/chunyuan-w/xfold.git
cd xfold
git checkout pt211_dev
```

---

## 3. Cache the data pipeline once (MSA search)

Run `run_pt211.sh` from the sglang repo root (`/workspace/sglang`), where the
script and its `env.sh` live.

The MSA search result is independent of padding, so run the data pipeline once
per input and reuse the cached `*_data.json` for all later inference runs.

```sh
INPUT_NAME=lrp2_4655.txt \
RUN_DATA_PIPELINE=True \
RUN_INFERENCE=False \
bash run_pt211.sh
```

Repeat for each input you intend to benchmark (`piezo2_2752.txt`,
`reelin_3469.txt`, ...).

---

## 4. Run inference with sgl-kernel

`run_pt211.sh` selects each AF3 module's backend via env vars. Each accepts
`cpp` (default) or `sgl`:

| Env var                            | Module                          |
| ---------------------------------- | ------------------------------- |
| `AF3_GRID_SELF_ATTENTION_IMPL`     | grid self-attention             |
| `AF3_SELF_ATTENTION_IMPL`          | single self-attention           |
| `AF3_TRIANGLE_MULTIPLICATION_IMPL` | triangle multiplication         |
| `AF3_GATED_LINEAR_UNIT_IMPL`       | gated linear unit               |

Full all-sgl run:

```sh
export OUTPUT_DIR=/pt211/workspace/outputs_debug
export INPUT_NAME=lrp2_4655.txt

# kernel impl selection — all SGL
export AF3_GRID_SELF_ATTENTION_IMPL=sgl
export AF3_TRIANGLE_MULTIPLICATION_IMPL=sgl
export AF3_GATED_LINEAR_UNIT_IMPL=sgl
export AF3_SELF_ATTENTION_IMPL=sgl

# run control
export PAD_TO_BUCKETS=False
export RUN_DATA_PIPELINE=False   # set True the first time you run a new input
export RUN_INFERENCE=True

bash run_pt211.sh
```

Or as a one-liner:

```sh
OUTPUT_DIR=/pt211/workspace/outputs_debug \
INPUT_NAME=lrp2_4655.txt \
AF3_GRID_SELF_ATTENTION_IMPL=sgl \
AF3_TRIANGLE_MULTIPLICATION_IMPL=sgl \
AF3_GATED_LINEAR_UNIT_IMPL=sgl \
AF3_SELF_ATTENTION_IMPL=sgl \
PAD_TO_BUCKETS=False \
RUN_DATA_PIPELINE=False \
RUN_INFERENCE=True \
bash run_pt211.sh
```

To get the `cpp` baseline for comparison, leave the four `AF3_*_IMPL` vars unset and rerun.

---
