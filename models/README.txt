Model history
=============
models/beacon_cnn.onnx  - ACTIVE: v2, trained on data/beacon_dataset_v2.npz (harmonic decoys,
                          turbulence up to 3, current beacon waveform). Promoted after a
                          40-seed closed-loop test: time locked 72.0% vs 71.5% (v1),
                          false-lock runs 2 vs 3; offline hybrid accuracy 91.6% vs 89.6%.
models/v1/               - previous model (dataset v1). To go back, set
                          identification.model_path: models/v1/beacon_cnn.onnx
models/candidate/        - output folder of tools/train_classifier.py (never used directly)
