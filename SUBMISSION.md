# Submission

**Repository:** [github.com/JoeIndyGit/14-Linear-to-MoE](https://github.com/JoeIndyGit/14-Linear-to-MoE)

**Notebook:** [Open in Colab](https://colab.research.google.com/github/JoeIndyGit/14-Linear-to-MoE/blob/main/linear_to_moe.ipynb)

## Summary to submit

I trained a literal 650-parameter linear classifier on handwritten digits and
converted its learned weights into four independent experts with top-2 routing.
Conversion preserves the logits to floating-point precision. Across five
initialization seeds and three stratified splits (15 paired cases), continued
MoE training reduces mean train CE from 0.172481 to
0.012547 and mean validation CE from
0.222265 to 0.099832.
Both losses decrease in every case. Final mean test accuracy is
97.28%.

The repository includes a notebook executed in a fresh Jupyter kernel,
all raw logs and resumable checkpoints, equal-step and projection-budget
linear controls, a scratch MoE, a balancing ablation, class routing/entropy
and gradient diagnostics, and automated verification. Accuracy advantages
vary, and all baselines and unfavorable outcomes are retained.

## Project entry points

1. [Reproducibility guide](docs/REPRODUCIBILITY.md)
2. [Executed notebook](linear_to_moe.ipynb)
3. [Full report](docs/EXPERIMENT.md)
4. [Raw comparison CSV](results/comparison.csv)

The archive contains the repository contents in `linear-to-moe/`. The GitHub
repository uses `main`; Colab and CI links point to that exact repository.
