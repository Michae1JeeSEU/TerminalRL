#!/usr/bin/env bash

set -u
set -o pipefail

mode="${1:-}"
repo="/home/user/projects/terminal"
python_bin="/home/user/envs/terminal_env/bin/python"
module="terminal_marl.training.run_v4_2_p3_controlled_emergency_admission_v5"
root="/home/user/training_outputs/terminal/p3_controlled_emergency_admission_v5_20260925"
control="${root}/_control"
checkpoint="/home/user/training_outputs/terminal/random_soc_gate_repair_v1_20260925/fit846401/random_soc_gate_repair_v1.pt"
expected_sha256="e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353"

case "${mode}" in
  smoke)
    stage="smoke"
    stage_name="V5_SMOKE"
    run_name="smoke849000"
    output="${root}/smoke"
    entry_report=""
    ;;
  entry)
    stage="entry"
    stage_name="V5_ENTRY"
    run_name="entry"
    output="${root}/entry"
    entry_report=""
    ;;
  pilot)
    stage="pilot"
    stage_name="V5_PILOT"
    run_name="pilot"
    output="${root}/pilot"
    entry_report="${root}/entry/manifest.json"
    ;;
  *)
    printf 'usage: %s smoke|entry|pilot\n' "$0" >&2
    exit 2
    ;;
esac

log="${control}/${run_name}.stdout.log"
status="${control}/${run_name}.status"
pid_file="${control}/${run_name}.pid"
exit_file="${control}/${run_name}.exitcode"
command_file="${control}/${run_name}.command"

if [[ -e "${output}" || -e "${log}" || -e "${status}" || \
      -e "${pid_file}" || -e "${exit_file}" || -e "${command_file}" ]]; then
  printf 'refusing existing per-stage output/control path: %s\n' "${run_name}" >&2
  exit 2
fi

if [[ "${stage}" == "pilot" && ! -f "${entry_report}" ]]; then
  printf 'pilot entry report missing: %s\n' "${entry_report}" >&2
  exit 1
fi

if [[ ! -f "${checkpoint}" ]]; then
  printf 'checkpoint missing: %s\n' "${checkpoint}" >&2
  exit 1
fi
actual_sha256="$(sha256sum "${checkpoint}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
  printf 'checkpoint SHA mismatch: %s (%s)\n' "${checkpoint}" "${actual_sha256}" >&2
  exit 1
fi

if ! mkdir -p "${control}"; then
  printf 'unable to create shared control directory: %s\n' "${control}" >&2
  exit 1
fi

command_text="cd ${repo} && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${python_bin} -m ${module} --stage ${stage} --checkpoint ${checkpoint} --checkpoint-kind repair --output ${output}"
if [[ "${stage}" == "pilot" ]]; then
  command_text="${command_text} --entry-report ${entry_report}"
fi
printf '%s\n' "${command_text}" > "${command_file}"
printf 'STARTING\n' > "${status}"

(
  cd "${repo}" || exit 1
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  if [[ -n "${entry_report}" ]]; then
    exec "${python_bin}" -m "${module}" \
      --stage "${stage}" \
      --checkpoint "${checkpoint}" \
      --checkpoint-kind repair \
      --output "${output}" \
      --entry-report "${entry_report}"
  else
    exec "${python_bin}" -m "${module}" \
      --stage "${stage}" \
      --checkpoint "${checkpoint}" \
      --checkpoint-kind repair \
      --output "${output}"
  fi
) > "${log}" 2>&1 &
pid=$!
printf '%s\n' "${pid}" > "${pid_file}"
printf 'RUNNING\n' > "${status}"

set +e
wait "${pid}"
rc=$?
printf '%s\n' "${rc}" > "${exit_file}"
if [[ "${rc}" -eq 0 ]]; then
  printf 'COMPLETED\n' > "${status}"
else
  printf 'FAILED\n' > "${status}"
fi
exit "${rc}"
