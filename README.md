# Semantic Compression and Concept Regularization in Text-Based Explainable Recommendation

This is the official PyTorch implementation of our paper:

> **Semantic Compression and Concept Regularization in Text-Based Explainable Recommendation**

The release covers the four generation-based backbones (NRT, PETER, SEQUER, TF-SEQUER), semantic IDs from OPQ, RQ-KMeans (Faiss residual quantization), and RQ-VAE, and sparse-autoencoder concept regularization.

## Dependencies

```shell
conda create -n <env_name> python=3.12
conda activate <env_name>
pip install -r requirements.txt

# Faiss (OPQ and RQ-KMeans semantic IDs) is only distributed through conda
conda install -c pytorch -c nvidia -c conda-forge faiss-gpu=1.13.0
# The faiss install replaces numpy with a newer build (incompatible with scipy/numba); restore the pinned one
pip install --force-reinstall --no-deps numpy==2.2.6
python -c "import stanza; stanza.download('en')"
python -c "import nltk; nltk.download('stopwords')"
```

The same environment runs vLLM (user profiles), sentence-transformers, SAE pretraining, semantic-ID training, and the recommenders.

## Datasets

Amazon datasets can be downloaded with `download_datasets.py`. The paper uses Beauty and Cellphones.

```shell
python download_datasets.py --dataset-names=Amazon_Beauty
# python download_datasets.py --dataset-names=Amazon_Beauty,Amazon_Cell_Phones_and_Accessories
```

For Yelp 2020, we used the processed version available at [this link](https://drive.google.com/file/d/1Xmn0RVx1bXRNkNNeSBON-8WoIBE1KytC/view?usp=sharing). The metadata must be downloaded separately from [Kaggle](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/download?datasetVersionNumber=2); the business metadata file should be placed at `data/Yelp/orig/yelp_academic_dataset_business.json`.

Datasets need to be processed via [Sentires](https://github.com/lileipisces/Sentires-Guide) to extract the item feature set. This item feature set will be loaded and used to extract explanation sentences from reviews, as well as evaluate feature-level explainability metrics.

Explanation extraction and preprocessing live under `nlp_xrs/data_preparation/`. Dataset-specific settings are in `config/data_preparation/`.

```shell
python prepare_data.py --action=extract_exps --datasets=beauty
# python prepare_data.py --action=extract_exps --datasets=beauty,cell,yelp

python prepare_data.py --action=preprocess_split --datasets=beauty
# python prepare_data.py --action=preprocess_split --datasets=beauty,cell,yelp
```

User profiles (taste and technical) are generated with vLLM by `prepare_ui_llm_desc.py` before user-side SID training. The paper uses `Qwen/Qwen2.5-32B-Instruct-AWQ` with the default generation settings, which match the user tokenizer configs:

```shell
python prepare_ui_llm_desc.py --entity-type user --desc-type taste --llm-model Qwen/Qwen2.5-32B-Instruct-AWQ --dataset beauty
python prepare_ui_llm_desc.py --entity-type user --desc-type technical --llm-model Qwen/Qwen2.5-32B-Instruct-AWQ --dataset beauty
```

The profiles are written to `data/<Dataset>/feat_only/<fold>/tokenizers/semantic_ids/`. The exact command is also printed when a required cache file is missing.

## Usage

`scripts/run_experiment.sh` runs a single experiment. `--config` is the YAML stem (or a nested id such as `ipm26/sid_xrs/...` for a recommender run).

```shell
# Item or user embeddings
bash scripts/run_experiment.sh embed --dataset beauty --config opq_qwen_3cb --mode item

# SAE concept dictionary (small / medium / large)
bash scripts/run_experiment.sh sae --dataset beauty --config medium --seed 1111

# Semantic IDs: OPQ, RQ-KMeans (rqfaiss_*), or RQ-VAE
bash scripts/run_experiment.sh sid --dataset beauty --config rqfaiss_qwen_3cb --mode item --seed 1111
bash scripts/run_experiment.sh sid --dataset yelp --config rqfaiss_qwen_user_profiles_16cb --mode user --seed 1111

# One recommender run. Raw-ID configs: nrt, peter, sequer, sequertextfusion
bash scripts/run_experiment.sh xrs --dataset beauty --config nrt --seed 1111
bash scripts/run_experiment.sh xrs --dataset beauty --config ipm26/sid_xrs/nrt-sid_agg__opq_qwen_3cb --seed 1111
```

Several runs can be dispatched across GPUs with `run_experiments.py`:

```shell
python run_experiments.py \
    --datasets beauty cell yelp \
    --configs nrt peter sequer sequertextfusion \
    --seeds 1111 24 53 91 126 \
    --gpus 0
```

Paper configurations are under `config/` (raw backbones), `config/ipm26/sid_xrs/` (SID, user-SID, and SAE ablations), `config/tokenizers/`, and `config/sae/`. `config/default/` holds shared data, model, and training defaults.


Recommender runs write metrics to `results/`.

> **Note on RPG and TIGER.** The sequential-recommendation comparison uses an external codebase. `map_to_memgengr.py` converts an SID mapping to the format expected by [MemGen-GR](https://github.com/Jamesding000/MemGen-GR/tree/main/genrec) combined with [RPG (KDD 2025)](https://github.com/facebookresearch/RPG_KDD2025).

## License

See [LICENSE](LICENSE) for details.
