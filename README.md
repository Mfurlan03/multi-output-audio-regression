# Multi-Output Audio Feature Regression

This project explores how well musical attributes can be predicted directly from full-length audio. I built an end-to-end pipeline that takes raw audio files, extracts features, and trains models to predict 11 continuous musical attributes, ranging from energy and valence to tempo and key.

The goal wasn’t just to get a low error score, but to understand which attributes are learnable, which ones dominate error, and where different model choices break down

## What this project does
- Predicts 11 musical attributes from raw audio
- Compares a strong, interpretable baseline (Random Forest) against a temporal neural model (GRU-based RNN)
- Emphasizes leakage-safe evaluation, per-feature error analysis, and reproducibility

Best result:
- Random Forest baseline (cross-validated MSE): 83.68
- GRU-based RNN (best test MSE): 3.97 after tuning

## Why this project
Musical attributes like danceability, energy, and valence are subjective and time-dependent, yet they’re widely used in recommendation and search systems.

This project asks:
- What can we reliably learn from audio alone?
- Which targets are stable versus inherently noisy?
- How do modeling choices affect different attributes, not just aggregate metrics?

Answering those questions matters more to me than just reporting a single headline score.

## Targets (11 features)
danceability, energy, valence, tempo, liveness, speechiness, instrumentalness, acousticness, loudness, time_signature, key

## Data pipeline
The pipeline was designed to be reproducible and leakage-safe from the start:
1. Collected labels and metadata from a large Spotify-derived dataset
2. Programmatically retrieved full-length audio and linked file paths to metadata
3.Preprocessed audio using Librosa (22,050 Hz) and extracted:
   - MFCCs, chroma, spectral contrast, tonnetz, zero-crossing rate
4. Stacked features into `[time_steps, feature_dims]` tensors
5. Save tensors as `.npy` files alongside aligned metadata
6. Split data by file into 70% train, 15% validation, 15% test to prevent leakage
   
To make the pipeline robust, I added timeouts, error handling during feature extraction, Unicode/path normalization, and checkpointed training to handle GPU interruptions.

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

## Results
- Random Forest cross-validated MSE: 83.68
- GRU RNN best Test MSE: **3.97**
- Per-feature MSE highlights
  - Low error: valence, liveness, energy
  - High error: tempo, loudness, key
- Training dynamics: rapid convergence in 10–12 epochs, checkpointing stabilized progress on Colab A100

## Reproducibility
- Leakage-safe split by file, 70 percent train, 15 percent validation, 15 percent test
- Versioned configs for each experiment
- Checkpointing by best validation loss

## Limitations
- Tempo variance dominates global MSE and can obscure improvements on other targets
- Feature scale differences not fully normalized in early iterations
- Data collection via third-party sources may reflect genre and popularity biases

## Ethical notes
- Respect copyright and platform terms when collecting audio
- Document data provenance and genre distribution biases
- Provide privacy-safe sample datasets for reproducibility

## Contact
- Michael Furlano, [linkedin.com/in/michaelfurlano](https://linkedin.com/in/michaelfurlano), [github.com/mfurlan03](https://github.com/mfurlan03), furlanomichael02@gmail.com
