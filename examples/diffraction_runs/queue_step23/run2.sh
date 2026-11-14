#!/usr/bin/env bash
# Second queue: V_mix3-5 again without early stopping ("b" runs, all 30 epochs). Waits for run.sh to finish.
set -u
cd "$(dirname "$0")/../.."
PY=/home/ubuntu/micromamba/envs/spotiflow/bin/python
Q=diffraction_runs/queue_step23
COMMON="--per-acq 1000 --eval-per-acq 500 --mix-sim --sim-repeat 12 --sim-val-to-train --lr 2e-4 --batch-size 16
        --epochs 30 --patience 8 --lr-patience 3 --select f1 --stop-patience 0"
stamp() { echo "$(date '+%F %T') $*" | tee -a $Q/queue.log; }
while pgrep -f "queue_step23/run.sh" > /dev/null; do sleep 60; done
stamp "run2: start"
train() {
    local name=$1; shift
    if [ -f diffraction_runs/ft_$name/model/thresholds.yaml ]; then stamp "$name already done, skipped"; return; fi
    stamp "start $name"
    $PY finetune_experimental.py --out diffraction_runs/ft_$name $COMMON "$@" > diffraction_runs/ft_$name.log 2>&1
    stamp "end $name (exit $?)"
}
train V_mix3b
train V_mix4b --weighted-labels
train V_mix5b --pseudo-from diffraction_runs/ft_V_mix2 --pseudo-thresh 0.55

models="V V_exp V_mix V_mix2"
for k in V_mix3 V_mix4 V_mix5 V_mix3b V_mix4b V_mix5b; do [ -f diffraction_runs/ft_$k/model/best.pt ] && models="$models $k"; done
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
stamp "queue2 done"
