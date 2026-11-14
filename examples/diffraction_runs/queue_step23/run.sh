#!/usr/bin/env bash
# Queue for improvement steps 2 and 3 (see 6_experimental_test_results.ipynb, section 8).
# Runs one after the other; a failed run does not stop the queue. Logs: diffraction_runs/<name>.log
#   bash diffraction_runs/queue_step23/run.sh        (from examples/)
set -u
cd "$(dirname "$0")/../.."
PY=/home/ubuntu/micromamba/envs/spotiflow/bin/python
Q=diffraction_runs/queue_step23
COMMON="--per-acq 1000 --eval-per-acq 500 --mix-sim --sim-repeat 12 --sim-val-to-train --lr 2e-4 --batch-size 16
        --epochs 30 --patience 8 --lr-patience 3 --select f1 --stop-patience 6 --min-epochs 10"

stamp() { echo "$(date '+%F %T') $*" | tee -a $Q/queue.log; }

train() {  # name, extra args
    local name=$1; shift
    if [ -f diffraction_runs/ft_$name/model/thresholds.yaml ]; then stamp "$name already done, skipped"; return; fi
    stamp "start $name"
    $PY finetune_experimental.py --out diffraction_runs/ft_$name $COMMON "$@" > diffraction_runs/ft_$name.log 2>&1
    stamp "end $name (exit $?)"
}

train V_mix3
train V_mix4 --weighted-labels
train V_mix5 --pseudo-from diffraction_runs/ft_V_mix2 --pseudo-thresh 0.55

models="V V_exp V_mix V_mix2"
for k in V_mix3 V_mix4 V_mix5; do [ -f diffraction_runs/ft_$k/model/best.pt ] && models="$models $k"; done
stamp "start eval ($models)"
$PY eval_experimental.py --models $models --out diffraction_runs/eval_exp2 > diffraction_runs/eval_exp2.log 2>&1
stamp "end eval (exit $?)"

stamp "start label check"
$PY check_exp_labels.py --no-sweep --models ${models#V } --thresholds diffraction_runs/eval_exp2/thresholds.json \
    --out diffraction_runs/eval_exp2/labels_check > diffraction_runs/eval_exp2_labels.log 2>&1
stamp "end label check (exit $?)"

stamp "start notebook section 8"
python3 $Q/add_sec8.py > $Q/notebook.log 2>&1
stamp "end notebook (exit $?)"
stamp "queue done"
