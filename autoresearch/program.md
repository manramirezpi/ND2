# ND2 Autoresearch Agent Instructions

You are an autonomous research agent. Your goal is to iteratively discover symbolic representations of physical/mathematical laws using the ND2 Monte Carlo Tree Search framework.

## Repository Mapping (Karpathy Template)
1. `prepare.py` — Generates synthetic test datasets. **Do not modify this** unless explicitly requested by the Human. Available dataset arguments:
   - `--dataset toy`: Simple polynomial ($x^2 + 2x$)
   - `--dataset harmonic`: Harmonic oscillator ($dv = -x$)
   - `--dataset legendre`: Phase 7 Multipole recurrence ($P_{l+1}$)

2. `train.py` — The single wrapper execution script you run. It accepts `--dataset` and internally sets up the vars and search boundaries. **You will edit hyperparameters, targets, and `SEED_EXPR` directly in this file.**

3. `program.md` — This file. It contains the human's CURRENT OBJECTIVE.

## How to Work
1. **Analyze**: Read the CURRENT OBJECTIVE below.
2. **Data Prep**: Generate correct data (e.g., `python prepare.py --dataset toy`).
3. **Hypothesize**: Modify `train.py` variables (e.g., tweaking `EPISODES` or creating a `SEED_EXPR` string to inject a known prior equation and escape local minima).
4. **Execute**: Run the search (e.g., `python train.py --dataset toy`).
5. **Iterate**: Look at the terminal Pareto front output. If it is not analytically perfect, tweak the `SEED_EXPR` or breadth params and re-run.

## CURRENT OBJECTIVE
**Familiarization Phase**
The human user wants to start small to get a feel for how the Autoresearch pipeline works.
**Task**: Successfully orchestrate the discovery of the `toy` and `harmonic` datasets. Verify that ND2 instantly snaps to $x^2 + 2x$ and $-x$. No seed expressions should be necessary for these since they are basic arithmetic (Tabula Rasa). 

Once you succeed here, we will transition to the Legendre Multpole experiment.
