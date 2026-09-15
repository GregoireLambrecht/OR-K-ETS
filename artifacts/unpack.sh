#!/bin/bash
# Restore the shipped snapshot into the folders the pipeline reads:
#   mm_warmstart_r3/                       step 2 output (market-maker warm start)
#   results_floor{20,30,40}/{choice,bau}/  step 3 output: 51 policy triples, progress.json,
#                                          convergence + deviator sheets
#   results_floor*/*/audit/<scenario>/     step 4 output: PMFG audit cells and paths.json
# Run from the operation_research/ folder. Existing files with the same names are overwritten.
set -e
cd "$(dirname "$0")/.."
tar xzf artifacts/mm_warmstart.tar.gz
for f in 20 30 40; do tar xzf artifacts/policies_floor$f.tar.gz; done
tar xzf artifacts/audit.tar.gz
echo "mm_warmstart_r3:  $(ls mm_warmstart_r3/*.eqx | wc -l) network"
for f in 20 30 40; do for m in choice bau; do
  printf "results_floor%s/%-7s %s policy triples, %s audit cells\n" $f $m \
    "$(ls results_floor$f/$m/policy_*_type0.eqx | wc -l)" \
    "$(ls results_floor$f/$m/audit/*/b*/k*/result.json | wc -l)"
done; done
