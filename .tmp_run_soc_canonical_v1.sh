#!/usr/bin/env bash

set -u
set -o pipefail

mode="${1:-}"
case "${mode}" in
  smoke)
    seed="848000"
    run_name="smoke848000"
    smoke_flag="--smoke"
    ;;
  848101)
    seed="848101"
    run_name="pair848101"
    smoke_flag=""
    ;;
  848102)
    seed="848102"
    run_name="pair848102"
    smoke_flag=""
    ;;
  *)
    printf 'usage: %s smoke|848101|848102\n' "$0" >&2
    exit 2
    ;;
esac

repo="/home/user/projects/terminal"
python_bin="/home/user/envs/terminal_env/bin/python"
root="/home/user/training_outputs/terminal/random_soc_canonical_reference_v1_20260925"
control="${root}/_control"
output="${root}/${run_name}"
log="${control}/${run_name}.stdout.log"
status="${control}/${run_name}.status"
pid_file="${control}/${run_name}.pid"
exit_file="${control}/${run_name}.exitcode"
command_file="${control}/${run_name}.command"
canonical_checkpoint="/home/user/training_outputs/terminal/p2_gate_repair_v3_anchored_20260919/gate_repair_v3_qualification_candidate.pt"
canonical_sha256="4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c"
candidate_checkpoint="/home/user/training_outputs/terminal/random_soc_gate_repair_v1_20260925/fit846401/random_soc_gate_repair_v1.pt"
candidate_sha256="e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353"

if [[ -e "${output}" || -e "${log}" || -e "${status}" || \
      -e "${pid_file}" || -e "${exit_file}" || -e "${command_file}" ]]; then
  printf 'refusing existing per-mode output/control path: %s\n' "${run_name}" >&2
  exit 2
fi

for pair in \
  "${canonical_checkpoint}|${canonical_sha256}" \
  "${candidate_checkpoint}|${candidate_sha256}"; do
  checkpoint="${pair%%|*}"
  expected_sha256="${pair##*|}"
  if [[ ! -f "${checkpoint}" ]]; then
    printf 'checkpoint missing: %s\n' "${checkpoint}" >&2
    exit 1
  fi
  actual_sha256="$(sha256sum "${checkpoint}" | awk '{print $1}')"
  if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    printf 'checkpoint SHA mismatch: %s (%s)\n' "${checkpoint}" "${actual_sha256}" >&2
    exit 1
  fi
done

if ! mkdir -p "${control}"; then
  printf 'unable to create control directory: %s\n' "${control}" >&2
  exit 1
fi

command_text="cd ${repo} && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${python_bin} -m terminal_marl.training.evaluate_random_soc_canonical_reference_v1 --canonical-checkpoint ${canonical_checkpoint} --candidate-checkpoint ${candidate_checkpoint} --output ${output} --seed ${seed}"
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
    "${python_bin}" -m terminal_marl.training.evaluate_random_soc_canonical_reference_v1 \
      --canonical-checkpoint "${canonical_checkpoint}" \
      --candidate-checkpoint "${candidate_checkpoint}" \
      --output "${output}" \
      --seed "${seed}" \
      "${smoke_flag}" > "${log}" 2>&1
  else
    "${python_bin}" -m terminal_marl.training.evaluate_random_soc_canonical_reference_v1 \
      --canonical-checkpoint "${canonical_checkpoint}" \
      --candidate-checkpoint "${candidate_checkpoint}" \
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
