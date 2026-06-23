# BKC: Running xfold (AF3) with sgl-kernel CPU kernels

This is the BKC for enabling the **sgl-kernel** CPU kernels on top of an existing
AlphaFold3 (AF3) Docker image.

The flow is: launch the AF3 container → build and install `sgl-kernel` → check
out `xfold` → cache the data pipeline once → run inference with the sgl kernels
selected via env vars.

> **Portability / mounts.** The helper scripts reference **fixed *container*
> paths** — `/data` (databases + params) and `/pt211` (working tree). To run on
> a different machine you change only the **host side** of each `docker -v`
> mount; keep the container-side targets `/data` and `/pt211` exactly as shown
> and everything works unchanged. If you genuinely must use different container
> targets, edit the path variables at the top of `run_pt211.sh` (see
> [Adapting to different paths](#adapting-to-different-paths)).

---

## 0. Prerequisites

- An existing AF3 Docker image (e.g. `xfold:cpu_3.0.0`) with the `cpp` kernels
  already installed.
- On the **host**: the AF3 databases + model params (the dir that becomes
  `/data`, containing `params/af3.bin`), and a writable working directory (the
  dir that becomes `/pt211`).
- Raw fold inputs, placed under `/pt211/workspace/inputs` inside the container
  (e.g. `amp_81.txt`, `piezo2_2752.txt`, `reelin_3469.txt`, `lrp2_4655.txt`).

---

## 1. Launch the container

```sh
docker run -it --rm \
  --privileged --net host --ipc host \
  -v <HOST_AF3_DATA>:/data \     # AF3 databases + model params  -> /data, /data/params
  -v <HOST_WORK_DIR>:/pt211 \    # persistent working tree       -> /pt211
  --name xfold \
  xfold:cpu_3.0.0
```

Replace only the **left** (host) side of each `-v`:

| Host path (yours) | Container path (fixed) | Holds |
| ----------------- | ---------------------- | ----- |
| `<HOST_AF3_DATA>` | `/data`                | AF3 databases; `params/af3.bin` |
| `<HOST_WORK_DIR>` | `/pt211`               | repos, inputs, outputs, logs |

> **`--rm` means the container filesystem is discarded on exit.** Clone repos and
> write all outputs **under `/pt211`** (a mount) so they persist. Anything written
> elsewhere in the container is lost when you exit.

Inside the container the working tree looks like this (created by the steps
below):

| Container path                | Purpose                                        |
| ----------------------------- | ---------------------------------------------- |
| `/data`, `/data/params`       | AF3 databases and model params                 |
| `/pt211/sglang`               | this repo — sgl-kernel build + `run_pt211.sh` + `env.sh` |
| `/pt211/xfold`                | xfold checkout (`run_alphafold.py`)            |
| `/pt211/workspace/inputs`     | raw fold inputs                                |
| `/pt211/workspace/outputs`    | processed JSON + results (MSA cache)           |
| `/pt211/workspace/log`        | run logs                                       |
| `/pt211/workspace/hmmer/bin`  | hmmer binaries (data pipeline)                 |

---

## 2. Build and install `sgl-kernel` (CPU)

```sh
cd /pt211
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
these drive the inference run (steps 4–5). `run_pt211.sh` sources the `env.sh`
sitting next to it; review `env.sh` and adjust only if your container mounts
differ from the defaults:

```sh
cd /pt211/sglang
# edit env.sh if needed: MODEL_DIR, OUTPUT_DIR, NCORES, etc.
```

---

## 3. Check out `xfold` with the sgl integration

`run_pt211.sh` runs xfold's `run_alphafold.py` (it `cd`s into `/pt211/xfold`
internally), so xfold must be checked out at exactly that path:

```sh
cd /pt211
git clone https://github.com/chunyuan-w/xfold.git
cd xfold
git checkout pt211_dev
```

---

## 4. Cache the data pipeline once (MSA search)

Run `run_pt211.sh` from the sglang repo root (`/pt211/sglang`), where the script
and its `env.sh` live.

The MSA search result is independent of padding, so run the data pipeline once
per input and reuse the cached `*_data.json` for all later inference runs.

```sh
cd /pt211/sglang
INPUT_NAME=lrp2_4655.txt \
RUN_DATA_PIPELINE=True \
RUN_INFERENCE=False \
bash run_pt211.sh
```

Repeat for each input you intend to benchmark (`piezo2_2752.txt`,
`reelin_3469.txt`, ...).

---

## 5. Run inference with sgl-kernel

`run_pt211.sh` selects each AF3 module's backend via env vars. Each accepts
`cpp` (default) or `sgl`:

| Env var                            | Module                          |
| ---------------------------------- | ------------------------------- |
| `AF3_GRID_SELF_ATTENTION_IMPL`     | grid self-attention             |
| `AF3_SELF_ATTENTION_IMPL`          | single self-attention           |
| `AF3_TRIANGLE_MULTIPLICATION_IMPL` | triangle multiplication         |
| `AF3_GATED_LINEAR_UNIT_IMPL`       | gated linear unit               |

Full all-sgl run (from `/pt211/sglang`):

```sh
cd /pt211/sglang

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
cd /pt211/sglang
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

## Adapting to different paths

The container-side mount targets are baked into `run_pt211.sh`. If you cannot use
`/data` and `/pt211` as the mount targets, change the host side of the `docker -v`
flags **and** update these variables at the top of `run_pt211.sh` to match:

| Variable in `run_pt211.sh` | Default            | Maps to mount |
| -------------------------- | ------------------ | ------------- |
| `cd /pt211/xfold`          | `/pt211/xfold`     | `/pt211`      |
| `DB_DIR`                   | `/data`            | `/data`       |
| `MODEL_DIR`                | `/data/params`     | `/data`       |
| `RAW_INPUT_DIR`            | `/pt211/workspace/inputs`  | `/pt211` |
| `*_OUTPUT_ROOT`            | `/pt211/workspace/outputs*`| `/pt211` |
| `LOG_DIR`                  | `/pt211/workspace/log`     | `/pt211` |
| `HMMER_BIN`                | `/pt211/workspace/hmmer/bin` | `/pt211` |

Keeping the targets as `/data` and `/pt211` is strongly recommended so the
scripts run unmodified.
