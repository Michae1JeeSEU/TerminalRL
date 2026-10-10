#!/usr/bin/env bash

set -u
set -o pipefail

mode="${1:-}"
case "${mode}" in
  smoke)
    seed="847000"
    run_name="smoke847000"
    smoke_flag="--smoke"
    ;;
  847101)
    seed="847101"
    run_name="pair847101"
    smoke_flag=""
    ;;
  847102)
    seed="847102"
    run_name="pair847102"
    smoke_flag=""
    ;;
  *)
    printf 'usage: %s smoke|847101|847102\n' "$0" >&2
    exit 2
    ;;
esac

repo="/home/user/projects/terminal"
python_bin="/home/user/envs/terminal_env/bin/python"
root="/home/user/training_outputs/terminal/random_soc_arbitration_repair_v1_20260925"
control="${root}/_control"
output="${root}/${run_name}"
log="${control}/${run_name}.stdout.log"
status="${control}/${run_name}.status"
pid_file="${control}/${run_name}.pid"
exit_file="${control}/${run_name}.exitcode"
command_file="${control}/${run_name}.command"
checkpoint="/home/user/training_outputs/terminal/random_soc_gate_repair_v1_20260925/fit846401/random_soc_gate_repair_v1.pt"
checkpoint_sha256="e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353"

if ! mkdir -p "${control}"; then
  printf 'unable to create control directory: %s\n' "${control}" >&2
  exit 1
fi

for path in "${output}" "${log}" "${status}" "${pid_file}" "${exit_file}" "${command_file}"; do
  if [[ -e "${path}" ]]; then
    printf 'refusing existing path: %s\n' "${path}" >&2
    exit 2
  fi
done

command_text="cd ${repo} && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${python_bin} -m terminal_marl.training.evaluate_random_soc_arbitration_repair_v1 --checkpoint ${checkpoint} --checkpoint-sha256 ${checkpoint_sha256} --output ${output} --seed ${seed}"
if [[ -n "${smoke_flag}" ]]; then
  command_text="${command_text} ${smoke_flag}"
fi
printf '%s\n' "${command_text}" > "${command_file}"
printf 'STARTING\n' > "${status}"

run_python() {
  local rc
  cd "${repo}" || return 1
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  if [[ -n "${smoke_flag}" ]]; then
    "${python_bin}" -m terminal_marl.training.evaluate_random_soc_arbitration_repair_v1 \
      --checkpoint "${checkpoint}" \
      --checkpoint-sha256 "${checkpoint_sha256}" \
      --output "${output}" \
      --seed "${seed}" \
      "${smoke_flag}" > "${log}" 2>&1
  else
    "${python_bin}" -m terminal_marl.training.evaluate_random_soc_arbitration_repair_v1 \
      --checkpoint "${checkpoint}" \
      --checkpoint-sha256 "${checkpoint_sha256}" \
      --output "${output}" \
      --seed "${seed}" > "${log}" 2>&1
  fi
  rc=$?
  printf '%s\n' "${rc}" > "${exit_file}"
  if [[ "${rc}" -eq 0 ]]; then
    printf 'COMPLETED\n' > "${status}"
  else
    printf 'FAILED\n' > "${status}"
  fi
  return "${rc}"
}

run_python &
pid=$!
printf '%s\n' "${pid}" > "${pid_file}"
printf 'RUNNING\n' > "${status}"
wait "${pid}"
exit $?
