# External Toxic Drug Analysis

This note summarizes what we found from `data/toxic_drugs_for_testing.csv` and how that should change the project strategy.

## Quick Findings

Dataset size:

- total rows: `2872`
- valid SMILES: `2869`

Dominant chemistry patterns in the toxic-drug set:

- aromatic compounds: `2140`
- fused or polycyclic compounds with at least 2 rings: `1811`
- molecules with H-bond donors: `1772`
- high lipophilicity with `MolLogP > 3`: `1503`
- halogenated compounds: `845`
- heteroaromatic compounds: `743`
- high polarity with `TPSA > 90`: `548`
- very high lipophilicity with `MolLogP > 5`: `438`
- heavily halogenated compounds with at least 3 halogens: `310`

Most common structural alerts in the toxic-drug set:

- aniline-like aromatic amine: `509`
- Michael acceptor: `316`
- aromatic nitro group: `177`
- hydrazine or hydrazone motif: `76`
- sulfonamide: `61`
- aromatic sulfonamide: `47`
- aldehyde: `38`
- epoxide: `29`

Descriptor summary:

- mean molecular weight: about `313.9`
- mean `MolLogP`: about `3.05`
- mean `TPSA`: about `62.36`
- median ring count: `2`
- median aromatic rings: `1`

## What This Means

The outside toxic-drug set is not mainly a simple "more toxic Tox21" version of the training set.

It is enriched for:

- reactive toxicophores
- aromatic amines
- nitro aromatics
- electrophilic motifs
- lipophilic halogenated scaffolds
- fused aromatic and polycyclic systems

That explains why the current pipeline struggles on outside data:

- Tox21 labels do not fully represent these real-world toxicity mechanisms
- the current ML core is still centered on assay endpoints, not broad drug toxicity outcomes
- outside toxic drugs contain many chemistry liabilities that need stronger explicit modeling
- mean-risk aggregation under-calls compounds with one strong liability mechanism

This is not just classic underfitting.

It is a mix of:

- task mismatch
- insufficient external label coverage
- incomplete toxicophore coverage
- weak external validation discipline

## Revised Strategy

## 1. Stop treating outside failure as only a tuning problem

Do not assume that more trees or more notebook tuning will solve this by itself.

The main issue is that the outside toxic-drug set contains chemistry patterns that are only partially represented in Tox21 supervision.

## 2. Make external toxic drugs a first-class evaluation set

Use `toxic_drugs_for_testing.csv` as a required external stress test.

For every model revision, report:

- verdict counts
- `Tox21 Mean Risk`
- `Tox21 Max Risk`
- structural alert coverage
- applicability-domain distribution
- top false negatives

This set should be part of the standard evaluation loop, not a one-off spot check.

## 3. Expand toxicophore coverage deliberately

The external set strongly supports keeping and growing a structural alert layer.

Prioritize alerts for:

- aromatic amines and aniline-like motifs
- nitro aromatics
- Michael acceptors
- hydrazines and hydrazones
- epoxides
- aldehydes
- isocyanates
- acyl halides
- halogen-rich aromatic systems

This layer should remain separate from the Tox21 model score so it can catch real-world liabilities that are weak in the assay model.

## 4. Use max-risk and agreement logic more heavily than mean-risk

For outside data, the mean over all endpoints is too forgiving.

Prefer:

- `max_endpoint_risk`
- structural alert severity
- agreement between endpoint signal and alert signal
- OOD penalty

The mean score can stay in the UI, but it should not dominate the final verdict.

## 5. Add stronger out-of-domain discipline

If a molecule is far from the Tox21/ZINC support space, the system should not act confident.

Outside-data policy should be:

- mark `Edge` or `Out-of-domain` clearly
- lower confidence
- avoid strong safety claims on unfamiliar chemistry

## 6. Add a broader supervised toxicity dataset next

This is the biggest strategy change.

If the goal is general drug-toxicity prediction, Tox21 alone is not enough.

Next dataset candidates should target real drug toxicity outcomes such as:

- DILI / hepatotoxicity
- hERG / cardiotoxicity
- mutagenicity
- acute toxicity

The best next step is not to replace Tox21, but to add another supervised dataset aligned with real toxic drug outcomes.

## 7. Reframe the model claim

Until a broader external dataset is added, present the system as:

- a Tox21 endpoint-risk model
- plus a structural-liability and applicability-domain layer

Do not frame it as a universal toxicity predictor.

## Practical Next Steps

1. Keep `toxic_drugs_for_testing.csv` in the permanent evaluation workflow.
2. Expand the alert library based on the dominant motifs above.
3. Make outside-data summary reporting automatic.
4. Add one broader supervised toxicity dataset.
5. Recalibrate the final verdict around max-risk, alerts, and OOD, not just mean-risk.

## Bottom Line

The outside toxic-drug set shows that the current model is missing broad real-world toxicity patterns, especially reactive and aromatic liability motifs.

So the strategy revision is:

- evaluate on external toxic drugs every time
- widen toxicophore coverage
- trust mean-risk less
- penalize OOD predictions harder
- add broader toxicity labels beyond Tox21
