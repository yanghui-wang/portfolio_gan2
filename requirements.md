# Requirements (Training + LLM Scoring)

This file lists the Python packages needed across both parts of this project:

- Portfolio GAN training/evaluation in `portfolio_gan2`
- LLM news scoring scripts in `LLM news scoring`

## Core Packages

```text
numpy>=1.26
pandas>=2.2
pyyaml>=6.0
python-dotenv>=1.0
pyarrow>=15.0
scikit-learn>=1.4
torch>=2.2
torchvision>=0.17
torchaudio>=2.2
tqdm>=4.66
tensorboard>=2.16
```

## LLM Scoring Packages

```text
openai>=1.0
datasets>=2.18
```

## Testing

```text
pytest>=8.0
```

## Recommended Install Command

Run in your activated environment:

```bash
python -m pip install \
  numpy>=1.26 pandas>=2.2 pyyaml>=6.0 python-dotenv>=1.0 pyarrow>=15.0 \
  scikit-learn>=1.4 torch>=2.2 torchvision>=0.17 torchaudio>=2.2 \
  tqdm>=4.66 tensorboard>=2.16 openai>=1.0 datasets>=2.18 pytest>=8.0
```

## Notes

- The import name is `yaml`, but the package name is `pyyaml`.
- If you already use `environment.yml`, this markdown file is a checklist/reference for dependency completeness.
