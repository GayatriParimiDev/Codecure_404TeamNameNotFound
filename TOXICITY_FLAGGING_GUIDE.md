# Toxicity Flagging Guide

This file describes what the current app will flag as toxic.

Important:
- This is the app's current decision logic.
- It is not a universal toxicology definition.
- The model is a Tox21 assay scorer plus a structural alert layer.

## How The App Flags Risk

The app now uses three signals:

1. `Tox21 Mean Risk`
   - Mean probability across the 12 assay endpoints.

2. `Structural Alert Score`
   - Rule-based chemistry alerts.
   - Current alerts focus on sulfonamide-like motifs.

3. `Final Verdict`
   - Composite score computed as:
   - `max(Tox21 Mean Risk, Structural Alert Score)`

The final band is then assigned from that composite score:

- `Low` if composite risk < 0.25
- `Moderate` if composite risk >= 0.25 and < 0.50
- `Elevated` if composite risk >= 0.50 and < 0.75
- `High` if composite risk >= 0.75

## What Will Be Flagged

### 1. Sulfonamide-like scaffolds

These are currently the strongest rule-based flags in the app.

Examples:

- Acetazolamide
  - SMILES: `CC(=O)Nc1nnc(S(N)(=O)=O)s1`
  - Expected final verdict: `Elevated`

- Methazolamide
  - SMILES: `CC(=O)N1/C(=N\\N=C1/C)S(N)(=O)=O`
  - Expected final verdict: `Elevated`

- Sulfanilamide
  - SMILES: `Nc1ccc(S(N)(=O)=O)cc1`
  - Expected final verdict: `Elevated`

- Benzothiazole-2-sulfonamide
  - SMILES: `NS(=O)(=O)c1nc2ccccc2s1`
  - Expected final verdict: `Elevated`

### 2. Molecules with high Tox21 endpoint risk

If the model assigns a high enough probability to one or more assay endpoints, the final verdict can rise even without a structural alert.

This includes molecules that:

- show strong activity across multiple Tox21 assays
- have a high single-endpoint risk and a higher mean score

## What May Still Show Low Risk

Some compounds are known toxicants in real-world toxicology, but the app may still show `Low` if:

- they do not match a current structural alert
- the Tox21 assay model does not strongly activate on them
- their toxicity mechanism is outside the current assay panel

Example:

- Nitrobenzene
  - SMILES: `C1=CC=C(C=C1)[N+](=O)[O-]`
  - It may still score low in this app if the current model does not recognize its toxicity mode.

## Current Alert Coverage

The current rule-based alert layer is intentionally narrow.

It currently emphasizes:

- sulfonamide
- aromatic sulfonamide
- heteroaromatic sulfonamide

That means the app is conservative for these motifs, but it does not yet include all toxicophore classes.

## Practical Interpretation

Use the app like this:

- `Low` means the Tox21 model and current structural alerts do not show a strong signal.
- `Moderate` means there is some evidence of concern.
- `Elevated` means the molecule should be reviewed carefully.
- `High` means the current app sees a strong toxicity signal.

For chemistry review, treat the structural alert as a hard warning, even if the Tox21 mean score is low.
