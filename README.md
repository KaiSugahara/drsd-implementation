# Towards Dynamic Relationship Schema Discovery for Complementary News Recommendation

This repository contains the implementation of the paper:

> Kai Sugahara. **Towards Dynamic Relationship Schema Discovery for Complementary News Recommendation.**
> In *Proceedings of the 20th ACM Conference on Recommender Systems (RecSys '26)*, September 27–October 2, 2026.
> DOI: [10.1145/3773078.3841261](https://doi.org/10.1145/3773078.3841261)

## Repository Structure

```
drsd/                   Core library
├── pipeline.py          Baseline / DRSD pipelines that orchestrate a full run
├── component/           Individual pipeline steps (sampling, annotation, training, pruning, ...)
├── trainer/              LightGBM-based relationship and CTR trainers
├── reader/                MIND-small dataset reader
├── schema.py             Relationship schema data structures
├── llm.py                 Gemini client used for hypothesis proposal / annotation
└── evaluation.py          Recommendation and CTR metrics

notebook/
├── preprocess/            Feature engineering notebooks (run once)
├── experiment/            Baseline and DRSD experiment notebooks
└── analyze/                Notebook for comparing MLflow results

.devcontainer/            Dev container definitions (GPU / CPU)
compose.yaml               Docker Compose service definition
Makefile                    Shortcuts for running notebooks
```

## Setup

### Prerequisites

- Docker and Docker Compose
- For the default (`cuda13`) dev container: an NVIDIA GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed. Use `.devcontainer/cpu` instead if no GPU is available.
- A [MIND-small](https://msnews.github.io/) dataset download
- A [Gemini API key](https://ai.google.dev/) for hypothesis proposal and annotation

### 1. Configure environment variables

```bash
cp .env.example .env
```

Fill in `.env`:

| Variable | Description |
| --- | --- |
| `GOOGLE_GENAI_API_KEY` | Gemini API key used by `drsd.llm.LLMClient` |
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI. Use `http://127.0.0.1:8080` to use the local server started automatically inside the dev container, or point to a remote server |
| `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` | Only required if the MLflow server requires authentication |

### 2. Download the dataset

Download [MIND-small](https://msnews.github.io/) and place it under `dataset/` as follows:

```
dataset/
├── MINDsmall_train/
│   ├── behaviors.tsv
│   └── news.tsv
└── MINDsmall_dev/
    ├── behaviors.tsv
    └── news.tsv
```

### 3. (Optional) Open the dev container

For interactive development (e.g. editing notebooks by hand), open the repository in VS Code and reopen it in the `cuda13` (or `cpu`) dev container. This builds the image, installs dependencies with Poetry, and starts a local MLflow server (`poetry run mlflow server --host 127.0.0.1 --port 8080`) that `MLFLOW_TRACKING_URI=http://127.0.0.1:8080` connects to.

## Usage

The commands below are run from the **host** — each `make` target invokes `docker compose run` itself, building the image on first use:

```bash
# 1. Feature engineering (writes parquet files under processed/)
make run_preprocess

# 2. Simple CTR baseline (no schema)
make run_baseline

# 3. Dynamic Relationship Schema Discovery (proposed method)
make run_proposal
```

`run_baseline` and `run_proposal` execute their notebook (`notebook/experiment/00_BASELINE.ipynb`, `notebook/experiment/01_PROPOSAL.ipynb`) end-to-end in a detached container and log metrics, prompts, and the discovered schema to MLflow. Make sure an MLflow server reachable at `MLFLOW_TRACKING_URI` is already running before starting them (e.g. via the dev container's `postAttachCommand`, or `docker compose run -d dev poetry run mlflow server --host 0.0.0.0 --port 8080`).

To compare results, open `notebook/analyze/00_METRICS_AND_HEATMAP.ipynb`, set the MLflow run IDs of your baseline/proposal runs, and run the notebook.

## Citation

```bibtex
@inproceedings{sugahara2026drsd,
  author    = {Sugahara, Kai},
  title     = {Towards Dynamic Relationship Schema Discovery for Complementary News Recommendation},
  booktitle = {Proceedings of the 20th ACM Conference on Recommender Systems (RecSys '26)},
  year      = {2026},
  doi       = {10.1145/3773078.3841261}
}
```
