#!/usr/bin/env bash
# Batch-size / resolution sweep on a single GPU.
# Reports steady-state iteration time and peak memory so throughput (videos/s)
# can be compared directly. Single-GPU on purpose: no DDP noise.
#
# usage: bench.sh <gpu_index>
set -uo pipefail

REPO="/home/ubuntu/inwdata/prithvi/git/vjepa2"
PY="${REPO}/venv/bin/python3"
GPU="${1:-1}"
cd "${REPO}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

run() {
  local bs=$1 crop=$2
  local tag="bs${bs}_crop${crop}"
  local cfg="workarea/bench_${tag}.yaml"
  sed -e "s|^  batch_size: .*|  batch_size: ${bs}|" \
      -e "s|^  crop_size: .*|  crop_size: ${crop}|" \
      -e "s|^  epochs: .*|  epochs: 1|" \
      -e "s|^  ipe: .*|  ipe: 12|" \
      -e "s|^  warmup: .*|  warmup: 0|" \
      -e "s|^folder: .*|folder: ${REPO}/workarea/runs/bench_${tag}|" \
      -e "s|^  enabled: true|  enabled: false|" \
      configs/train_2_1_face/faces-8d-384px-30f.yaml > "${cfg}"
  rm -rf "workarea/runs/bench_${tag}"
  timeout 900 "${PY}" -u -m app.main --fname "${cfg}" --devices "cuda:${GPU}" \
      > "workarea/logs/bench_${tag}.log" 2>&1
  # last logged iteration of the final epoch = steady state
  local line
  line=$(grep -a "log_stats" "workarea/logs/bench_${tag}.log" | tail -1)
  local it mem
  it=$(echo "$line"  | sed -n 's/.*\[iter: \([0-9.]*\) ms\].*/\1/p')
  mem=$(echo "$line" | sed -n 's/.*\[mem: \([0-9.e+]*\)\].*/\1/p')
  if [ -z "$it" ]; then
    if grep -qa "OutOfMemory" "workarea/logs/bench_${tag}.log"; then
      printf '%-16s %-10s %-12s %s\n' "${bs}" "${crop}" "OOM" "-"
    else
      printf '%-16s %-10s %-12s %s\n' "${bs}" "${crop}" "FAILED" "-"
    fi
  else
    "${PY}" - "$bs" "$it" "$mem" "$crop" <<'EOF'
import sys
bs, it, mem, crop = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
print(f"{bs:<16}{crop:<10}{mem/1024:<12.2f}{it/1000:<12.2f}{bs/(it/1000):<12.2f}")
EOF
  fi
  rm -rf "workarea/runs/bench_${tag}" "${cfg}"
}

printf '%-16s%-10s%-12s%-12s%-12s\n' "batch/gpu" "crop" "mem(GB)" "s/iter" "vids/s/gpu"
for spec in "8 384" "16 384" "24 384" "16 256" "32 256"; do
  run $spec
done
