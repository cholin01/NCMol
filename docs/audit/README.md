# Experiment audit (2026-10-29)

The current trainer selects checkpoints and schedules learning rates using validation
only. Test evaluation occurs after training. The summarizer reports all available
runs in fixed order; it does not select a best model by test score. Regression tests
cover both summary ordering and training independence from changed test labels.

## Historical evidence

`experiment_tree_20261029.json` records a read-only audit of six original datasets
and 76 checkpoints with their histories, configurations and SHA-256 hashes.
All 13 dataset/split combinations have zero exact canonical isomeric molecule
duplicates within or across partitions. Traditional ML, graph baseline and NCMol
partition memberships and labels agree. All 76 checkpoint epochs match the historical
validation-only selection rule, and saved scaling matches training-only statistics.
Historical logs nevertheless expose test metrics each epoch; manual decisions cannot
be certified independent of those observations.

An actual cross-task overlap was found in PAMPA-to-Caco-2 ER transfer (both variants).
For random splits, five target test molecules occur in source training and two in
source validation; three target validation molecules occur in source training.
For scaffold splits, one target validation molecule occurs in source training and
one in source validation. These results are not certified as unseen-molecule transfer.
They can be reported as cross-endpoint transfer with shared molecules. To evaluate
unseen-molecule transfer, source and target training must instead be rerun with
source training/selection excluding target holdouts.
No historical datasets, models or reported scores were changed by this audit.

`saved_results_20261029.json` reconciles 7,726 predictions across 136 evaluation groups.
All available reported metrics agree within 1e-5; 13 task/split evaluation sets have
matching molecular identities and labels across methods. This establishes result
consistency, not absence of training leakage.

Connectivity, scaffold and parent/tautomer comparisons are additional diagnostics,
not interchangeable identity definitions. RDKit warned that some tautomer enumerations
hit limits; those results are exploratory and not an exhaustive equivalence certificate.
No software audit proves undocumented human model choices or external-data provenance.

The audit scripts do not modify experimental inputs. Missing source evidence must
not be replaced by invented manifests or declared clean. Current initialization
checks reject absent provenance. The default molecule-disjoint policy excludes all
source training/selection overlap with target holdouts. The explicit cross-task
policy permits different-endpoint overlap, records task ancestry, and rejects
target-task exposure in any ancestor. Legacy checkpoints remain usable for inference.
