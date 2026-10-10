#!/usr/bin/env bash

set -euo pipefail

run_index="${1:-}"
if [[ "$#" -ne 1 || ! "$run_index" =~ ^[012]$ ]]; then
  printf 'usage: %s 0|1|2\n' "$0" >&2
  exit 2
fi

repo="/home/user/projects/terminal"
python_bin="/home/user/envs/terminal_env/bin/python"
module="terminal_marl.training.run_v4_2_p3_controlled_emergency_formal_v5"
root="/home/user/training_outputs/terminal/p3_controlled_emergency_formal_v5_restart1_20260925"
control="${root}/_control"
run_name="run${run_index}"
output="${root}/${run_name}"
admission_report="/home/user/training_outputs/terminal/p3_controlled_emergency_admission_v5_20260925/formal_admission_decision.json"
checkpoint="/home/user/training_outputs/terminal/random_soc_gate_repair_v1_20260925/fit846401/random_soc_gate_repair_v1.pt"
formal_protocol="${repo}/terminal_marl/configs/low_v4_2/p3_controlled_emergency_formal_v5.yaml"
expected_checkpoint_sha256="e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353"
expected_decision_sha256="d4f34955bd2230e5f531ab71f451d7eb538fbd6f16cc6bf05387ff0a673d2d8d"
expected_formal_protocol_sha256="e3f88e52a327c64621226631c1575cd48ffec4d93b98d505c561fe55b33b22b1"

command_file="${control}/${run_name}.command"
pid_file="${control}/${run_name}.pid"
launcher_pid_file="${control}/${run_name}.launcher.pid"
status_file="${control}/${run_name}.status"
exit_file="${control}/${run_name}.exitcode"
stdout_file="${control}/${run_name}.stdout.log"
stderr_file="${control}/${run_name}.stderr.log"

for path in "${output}" "${command_file}" "${pid_file}" "${launcher_pid_file}" \
            "${status_file}" "${exit_file}" "${stdout_file}" "${stderr_file}"; do
  if [[ -e "$path" ]]; then
    printf 'refusing existing per-run output/control path: %s\n' "$path" >&2
    exit 2
  fi
done

if [[ ! -x "$python_bin" ]]; then
  printf 'python executable missing: %s\n' "$python_bin" >&2
  exit 1
fi
if [[ ! -f "$admission_report" || ! -f "$checkpoint" || ! -f "$formal_protocol" ]]; then
  printf 'required binding file missing\n' >&2
  exit 1
fi

check_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    printf 'SHA mismatch: %s (%s)\n' "$path" "$actual" >&2
    exit 1
  fi
}

check_sha256 "$checkpoint" "$expected_checkpoint_sha256"
check_sha256 "$admission_report" "$expected_decision_sha256"
check_sha256 "$formal_protocol" "$expected_formal_protocol_sha256"

if ! mkdir -p "$control"; then
  printf 'unable to create shared control directory: %s\n' "$control" >&2
  exit 1
fi

command_text="cd ${repo} && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ${python_bin} -m ${module} --run-index ${run_index} --admission-report ${admission_report} --checkpoint ${checkpoint} --output ${output}"
printf '%s\n' "$command_text" > "$command_file"
printf 'STARTING\n' > "$status_file"

nohup bash -c '
set -u
repo="$1"
python_bin="$2"
module="$3"
run_index="$4"
admission_report="$5"
checkpoint="$6"
output="$7"
pid_file="$8"
status_file="$9"
exit_file="${10}"
stdout_file="${11}"
stderr_file="${12}"

cd "$repo" || {
  printf "FAILED\n" > "$status_file"
  printf "1\n" > "$exit_file"
  exit 1
}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$python_bin" -m "$module" \
  --run-index "$run_index" \
  --admission-report "$admission_report" \
  --checkpoint "$checkpoint" \
  --output "$output" > "$stdout_file" 2> "$stderr_file" &
child="$!"
printf "%s\n" "$child" > "$pid_file"
printf "RUNNING\n" > "$status_file"
wait "$child"
rc="$?"
printf "%s\n" "$rc" > "$exit_file"
if [[ "$rc" -eq 0 ]]; then
  printf "COMPLETED\n" > "$status_file"
else
  printf "FAILED\n" > "$status_file"
fi
exit "$rc"
' formal-v5-restart1-monitor "$repo" "$python_bin" "$module" "$run_index" \
  "$admission_report" "$checkpoint" "$output" "$pid_file" "$status_file" \
  "$exit_file" "$stdout_file" "$stderr_file" >/dev/null 2>&1 &
launcher_pid="$!"
printf '%s\n' "$launcher_pid" > "$launcher_pid_file"

for ((attempt=0; attempt<100; attempt++)); do
  if [[ -s "$pid_file" ]]; then
    printf 'launched %s python_pid=%s launcher_pid=%s\n' "$run_name" \
      "$(cat "$pid_file")" "$launcher_pid"
    exit 0
  fi
  sleep 0.1
done

printf 'launcher did not persist python pid: %s\n' "$run_name" >&2
exit 1
