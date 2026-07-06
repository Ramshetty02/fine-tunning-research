# Can You Compress Fine-Tuning Into a Vector?

**Five experiments on a free GPU: curriculum learning, catastrophic forgetting, sparse autoencoders, and activation steering — on a 0.5B model.**

This project started as a simple question — *does the order of training data matter when fine-tuning a small language model?* — and ended somewhere much stranger: more than half of a fine-tune's performance gain, compressed into a single 896-dimensional vector.

All experiments run on a free Google Colab T4 GPU. Model: Qwen2.5-0.5B (base and Instruct). Task: GSM8K grade-school math. Evaluation: exact-match accuracy on the same 200 held-out test problems in every round.

## Headline result

| Setup | GSM8K accuracy (200 problems) |
|---|---|
| Base model, untouched | 6.5% |
| Base + **one raw steering vector** (no training) | **21.5%** |
| Base + top-100 SAE features only | 17.0% |
| Full LoRA fine-tune (1 GPU-hour) | 34.0% |

A single mean-activation-difference vector, injected at layer 12 of the frozen base model, recovers **55%** of fine-tuning's gain. Filtering that vector through a home-trained sparse autoencoder — keeping only the ~100 most-shifted, human-inspectable features — still recovers **38%**.

## The five rounds

### Round 1 — Curriculum hypothesis (`curriculum_slm_experiment.ipynb`)
**Hypothesis:** easy→hard ordering beats shuffled data for small-model fine-tuning at fixed compute.
Fine-tuned Qwen2.5-0.5B-**Instruct** three times on the same 2,000 GSM8K problems, identical hyperparameters, only the order changed: shuffled / easy→hard / hard→easy. Difficulty proxy: number of calculation steps in the reference solution. A custom `OrderedTrainer` forces a sequential sampler (HF Trainer shuffles silently by default).

**Result:** every fine-tuned model scored *below* the untouched baseline (43.0% → 34.0 / 30.5 / 32.0%). Fine-tuning was net-harmful — **catastrophic forgetting**, measured accidentally. The Instruct model already knew GSM8K-style math better than GSM8K's own terse reference answers; training on them degraded it.

### Round 2 — Gentler training (`curriculum_slm_round2.ipynb`)
Two changes: learning rate 2e-4 → 2e-5, and prompt masking (loss on answer tokens only).

**Result:** still net-harmful, just less (43.0% baseline vs 36.0 / 33.5 / 29.0%). The damage was a property of the regime (re-training on known material), not the hyperparameters. Ordering still uninterpretable in a broken regime.

### Round 3 — The fair test (`curriculum_slm_round3.ipynb`)
Switched to the **base** (non-instruct) model, plain `Question:/Answer:` format, LR 1e-4, prompt masking. The base model scores 6.5% zero-shot — a true novice — so fine-tuning must help, and finally the ordering question can be asked fairly.

**Result:** fine-tuning worked massively (6.5% → ~33%, a 5x jump from 2,000 examples). And the orderings? **A 33.0% / B 33.5% / C 32.0% — statistically indistinguishable.** The curriculum hypothesis is dead at this scale: ordering by difficulty neither helps nor hurts. Just shuffle.

Bonus observation: base 6.5% vs Instruct 43.0% on the same problems is the value of instruction tuning, measured directly.

### Round 4 — Sparse autoencoder microscope (`sae_round4.ipynb`)
A scaled-down version of Anthropic-style interpretability. Captured layer-12 residual-stream activations from both the base and fine-tuned models on 300 held-out problems (~100k token vectors), trained one shared sparse autoencoder (896 → 7,168 features, L1-sparsity), and ranked features by how much fine-tuning shifted their mean activation.

**Result:** clean monosemantic features emerged — a division feature (fires on "12 / 3", "1000 / 2"), a totals/aggregation feature, a comparison feature ("7 more than"), a rates feature ("$/week * 50 weeks/year"), and a feature firing precisely on the "?\nAnswer:" boundary of our training format. Amplified features were dominated by arithmetic-procedure machinery; the most-suppressed feature fired on *formal algebra* ("dividing both sides by 11") — fine-tuning appears to have shifted the model from an algebraic register to GSM8K's step-by-step arithmetic dialect. All shifts were small (~5–10% relative), consistent with LoRA being a low-rank nudge.

### Round 5 — Steering: fine-tuning as a vector (`steering_round5.ipynb`)
Computed the steering vector = mean(fine-tuned − base activations) at layer 12 over 300 *training* problems (no test leakage). Injected `alpha * vector` into the frozen base model's layer-12 output during generation via a forward hook. Swept alpha; peak found at 8. Variant B: reconstructed the vector from only the top-K most-shifted SAE features (sparse, interpretable steering).

**Result:** the headline table above. One vector: 55% of the fine-tune's gain. One hundred nameable features: 38%.

## What this project establishes (and what it doesn't)

Established, within its scope:
1. Fine-tuning an instruction-tuned model on material it already knows can substantially degrade it — always benchmark the untouched baseline.
2. Curriculum ordering by difficulty makes no detectable difference for LoRA SFT at 0.5B scale on GSM8K.
3. A large fraction of what this fine-tune "did" is a low-dimensional, partially interpretable shift in activation space.

Not established: generality. One model, one task, one layer, one seed. The recovery fraction (55%) could differ at other scales, tasks, or layers. Alpha=8 was the edge of the sweep; the true peak may be slightly higher. Feature interpretations are human pattern-reading and should be held lightly. Related research (task vectors, activation steering, SAE interpretability) predates this project; the contribution here is the specific end-to-end chain and the quantified sparse-vs-raw comparison at hobby scale.

## Reproducing

Each notebook is self-contained: upload to Google Colab, set Runtime → T4 GPU, Run all. Rounds 4–5 automatically retrain the round-3 adapter if it isn't on disk. Rough runtimes: R1–R3 ~1.5–2h each, R4 ~40–60min, R5 ~1.5h. Kaggle also works (enable Internet in notebook settings; phone verification required for GPU).

Known environment gotcha: Colab's preinstalled `torchao` conflicts with recent `peft`. Every notebook uninstalls it in Cell 1; if you hit an import error after installs, restart the session and run again.

## Repository contents

```
curriculum_slm_experiment.ipynb   # Round 1 — Instruct model, LR 2e-4
curriculum_slm_round2.ipynb       # Round 2 — LR 2e-5 + prompt masking
curriculum_slm_round3.ipynb       # Round 3 — base model, fair test
sae_round4.ipynb                  # Round 4 — sparse autoencoder
steering_round5.ipynb             # Round 5 — steering vector
README.md
```

## Acknowledgments

Experimental designs were developed in collaboration with Claude (Anthropic), which proposed hypotheses and wrote the notebooks; all experiments were run, debugged, and verified by me, and every result in this README came off my GPU sessions. Claude's predictions went 0-for-3 against the experimental results — a small, satisfying datapoint on the difference between theory and measurement.

Built on: [Qwen2.5](https://huggingface.co/Qwen) (Alibaba), [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (OpenAI), Hugging Face `transformers`/`peft`/`datasets`. SAE methodology inspired by Anthropic's interpretability research ([Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)).