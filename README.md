# Multi-Output Audio Feature Regression
Predict 11 musical attributes from full-length audio using a baseline Random Forest and a GRU-based RNN. The pipeline covers data acquisition, Librosa-based preprocessing, leakage-safe splitting, model training, evaluation, and reporting.

- Baseline, Random Forest, cross-validated MSE: 83.68
- Primary, GRU RNN, best Test MSE: **3.97** after tuning
- Notable finding: features with low variability, for example valence, liveness, energy, regress well, tempo remains high-variance and dominates aggregate MSE

Michael Furlano, University of Toronto, Industrial Engineering, graduation May 2026

## Why this project
Musical attributes such as danceability, energy, and valence are subjective and time-dependent. We convert raw audio into feature tensors and learn to predict 11 continuous attributes, enabling objective comparisons, better search, and production-friendly scoring for music analysis.

## Targets (11 features)
danceability, energy, valence, tempo, liveness, speechiness, instrumentalness, acousticness, loudness, time_signature, key

## Data pipeline
1. Source labels and metadata from a large Spotify-derived dataset
2. Download full-length audio via programmatic retrieval, store file paths in the metadata CSV
3. Preprocess audio with Librosa at 22050 Hz and extract:
   - MFCCs, chroma, spectral contrast, tonnetz, zero-crossing rate
4. Stack and transpose features into 2D tensors shaped as `[time_steps, feature_dims]`
5. Save tensors to `.npy`, export aligned metadata CSV
6. Split by file into 70 percent train, 15 percent validation, 15 percent test, preventing leakage

Resilience features: timeouts and error handling during feature extraction, path normalization, Unicode normalization to remove accents and symbols, checkpointed training to handle GPU interruptions.

## Models
- **Baseline, Random Forest Regressor**
  - Aggregates per-song features by mean and standard deviation to create fixed-length vectors
  - `RandomizedSearchCV` across `n_estimators`, `max_depth`, `max_features`, `bootstrap`
  - Best CV MSE: 83.6792
  - Strengths: speechiness, valence, liveness; Weakness: tempo, loudness, key

- **Primary, GRU-based RNN in PyTorch**
  - Five stacked GRU layers, hidden size 256, dropout 0.3–0.5, layer norm, linear head to 11 outputs
  - Loss: aggregated Mean Squared Error across outputs
  - Experiments
    - Exp 1: hidden 256, lr 0.001, dropout 0.3, Test MSE ≈ 90.15
    - Exp 2: hidden 512, lr 0.0005, dropout 0.3, Test MSE ≈ 24.67
    - Exp 3: hidden 256, lr 0.001, dropout 0.5, Test MSE ≈ 3.97
    - **Best overall Test MSE: 3.97** 
  - Note: despite temporal modeling, the RNN initially underperformed the RF baseline on aggregate due to high-variance features, especially tempo. Further tuning achieved a significantly lower MSE.

## Key insights
- Aggregate MSE can hide per-feature strengths; per-feature MSE heatmaps reveal that tempo dominates error
- Strong regularization, dropout 0.5, reduced overfitting and improved RNN test MSE by about 4 percent (in early experiments)
- Leakage-safe splits, fixed seeds, and documented preprocessing are critical to fair comparison
- For production, combine a temporal model with feature-specific heads or loss weighting to address high-variance features

## Repo structure
- `src/`
  - `data_gen.py`, `data_utils.py`, `features_librosa.py`
  - `train_rf.py`, `train_gru.py`, `eval_metrics.py`, `plots.py`
- `notebooks/`
  - `01_eda.ipynb`, `02_baseline_rf.ipynb`, `03_gru_train_eval.ipynb`
- `data/`
  - `README.md` with pointers to sample data or generation scripts, keep large files out of Git
- `configs/`
  - `rf.yaml`, `gru_exp1.yaml`, `gru_exp2.yaml`, `gru_exp3.yaml`, `seeds.yaml`
- `docs/`
  - `mae_mse_plots`, `per_feature_mse_heatmap.png`, `loss_curves_exp1_3.png`, `dataset_schema.md`
- `requirements.txt`
- `.env.example`
- `LICENSE`, `.gitignore`, `README.md`

## Quickstart
1.  **Environment**
    -   Python 3.10+
    -   Create and activate a venv

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

2.  **Data**
    -   Place or generate sample audio and metadata
    -   Option A: use provided small sample in `data/sample`, run feature extraction
    -   Option B: point to your own audio and metadata CSV

    ```bash
    python src/features_librosa.py --csv data/metadata.csv --audio_root data/audio --out_root data/processed --sr 22050 --timeout_sec 120
    ```

3.  **Baseline, Random Forest**
    ```bash
    python src/train_rf.py --features data/processed/features_train.npy --labels data/processed/labels_train.csv --val_features data/processed/features_val.npy --val_labels data/processed/labels_val.csv --config configs/rf.yaml --out models/rf_baseline.joblib
    ```

4.  **GRU model**
    ```bash
    python src/train_gru.py --train_np data/processed/features_train.npy --train_labels data/processed/labels_train.csv --val_np data/processed/features_val.npy --val_labels data/processed/labels_val.csv --config configs/gru_exp3.yaml --out models/gru_exp3.pt
    ```

5.  **Evaluation and plots**
    ```bash
    python src/eval_metrics.py --test_np data/processed/features_test.npy --test_labels data/processed/labels_test.csv --model models/gru_exp3.pt --type gru --plots_out docs/
    ```

## Results
- Random Forest cross-validated MSE: 83.68
- GRU RNN best Test MSE: **3.97**
- Per-feature MSE highlights
  - Low error: valence, liveness, energy
  - High error: tempo, loudness, key
- Training dynamics: rapid convergence in 10–12 epochs, checkpointing stabilized progress on Colab A100

See `docs/per_feature_mse_heatmap.png` and `docs/loss_curves_exp1_3.png`.

## Reproducibility
- Fixed seeds in `configs/seeds.yaml`
- Leakage-safe split by file, 70 percent train, 15 percent validation, 15 percent test
- Versioned configs for each experiment
- Checkpointing by best validation loss
- Deterministic preprocessing with Librosa parameters recorded in `dataset_schema.md`

## Limitations
- Tempo variance dominates global MSE and can obscure improvements on other targets
- Feature scale differences not fully normalized in early iterations
- Data collection via third-party sources may reflect genre and popularity biases

## Roadmap
- Per-feature loss weighting or heteroscedastic regression to down-weight high-variance targets
- Multi-head architecture with specialized heads for tempo and key
- Spectrogram + CNN front-ends, or CNN-RNN hybrids for improved local-temporal capture
- Better scale normalization, label standardization, and calibration
- Active learning loops to target poorly performing feature regions

## Ethical notes
- Respect copyright and platform terms when collecting audio
- Document data provenance and genre distribution biases
- Provide privacy-safe sample datasets for reproducibility

## Credits
- Course: APS360, University of Toronto, Group 11
- Authors: Michael Furlano, Cassandra Mack, Adam Barhoush, Jad Al-Jawhari

## Contact
- Michael Furlano, [linkedin.com/in/michaelfurlano](https://linkedin.com/in/michaelfurlano), [github.com/mfurlan03](https://github.com/mfurlan03), furlanomichael02@gmail.com
