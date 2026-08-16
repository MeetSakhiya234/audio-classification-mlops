# Baseline Evaluation Results

## Model

- Model: `superb/wav2vec2-base-superb-ks`
- Architecture: Wav2Vec2
- Task: Speech command classification
- Number of classes: 12
- Sample rate: 16000 Hz

## Dataset

- Dataset: `google/speech_commands`
- Version: `v0.02`
- Evaluation split: Official test set
- Evaluation samples: 120
- Samples per class: 10

## Baseline Metrics

| Metric | Score |
|---|---:|
| Accuracy | 0.9750 |
| Macro F1 | 0.9753 |
| Macro Precision | 0.9785 |
| Macro Recall | 0.9750 |

## MLflow

- Experiment: `audio-classification`
- Run name: `baseline-wav2vec2-evaluation`
- Run ID: `c8bf9dd870aa40d6bfb9b3924b4fc45f`

## Notes

The baseline evaluation successfully completed using CPU inference.

The current development laptop has limited WSL memory, so further Wav2Vec2 optimization will be continued on a higher-resource machine.
