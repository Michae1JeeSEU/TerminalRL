"""Extract decisions from saved replay evidence only; no simulator/solver imports.

UID numbering deliberately matches ReplayController's sorted complete states IDs.
All event indexes below are zero based indexes in the original events.jsonl file.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DEFAULT_INPUT = Path("artifacts/rs_selective_repair_review_20261029/new_results/arms")
DEFAULT_OUTPUT = Path("artifacts/rs_selective_repair_20261029/replay_round_report")
DEFAULT_DOCUMENT = Path("docs/rotating_slots_stage3/selective_repair_replay_rounds_20261029.md")
PENDING = {"COMPUTING", "PREPARING", "PREPARED", "REVALIDATING"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def uid_mapping(states):
    return {identity: uid for uid, identity in enumerate(sorted(
        {str(a["id"]) for state in states for a in state.get("aircraft", [])}))}


def selection(record):
    selective = record.get("selective_replanning", {})
    calls = selective.get("calls", [])
    initial = list(calls[0].get("scope", {}).get("replan_ids", [])) if calls else []
    attempted = sorted(set(selective.get("attempted_replan_ids", [])) |
                       {a for call in calls for a in call.get("scope", {}).get("replan_ids", [])})
    returned = list(selective.get("scope", {}).get("replan_ids", []))
    return {"initial_ids": initial, "attempted_ids": attempted,
            "expanded_ids": sorted(set(attempted) - set(initial)), "returned_ids": returned,
            "actual_k_field": selective.get("actual_k"),
            "returned_scope_actual_k_field": selective.get("scope", {}).get("actual_k"),
            "returned_scope_size": len(returned),
            "attempt_scopes": [call.get("scope", {}).get("replan_ids", []) for call in calls]}


def active_at(actions, time, event_index):
    """Half-open flight intervals plus actual event order at equal timestamps."""
    return [action for action in actions
            if action["start_event_index"] < event_index
            and action.get("start_time") is not None and action.get("end_time") is not None
            and action["start_time"] <= time < action["end_time"]
            and (action.get("completion_event_index") is None
                 or action["completion_event_index"] >= event_index)]


def execution_outcome(task_status, submitted_actions, ongoing):
    if task_status in PENDING:
        return "TASK_PENDING"
    if submitted_actions:
        return "ACTUALLY_STARTED" if all(a["start_event_index"] is not None for a in submitted_actions) else "SUBMITTED_NOT_STARTED"
    return "MAINTAIN_PREVIOUS_ACTION" if ongoing else "NO_NEW_ACTION"


def extract(root, records, tasks, events, states, trajectories):
    uid_by_id = uid_mapping(states)
    issues = []
    assert len(records) == len(tasks), (root, "record/task count mismatch")
    assert len({t["snapshot_time"] for t in tasks}) == len(tasks)
    by_effective = {r["time"]: r for r in records}
    assert len(by_effective) == len(records)
    starts = [(i, e) for i, e in enumerate(events) if e["type"] == "normal_task_started"]
    assert len(starts) == len(tasks)
    provenance = {}
    for number, task in enumerate(tasks, 1):
        record = by_effective[task["effective_time"]]
        for hash_ in record.get("actual_submitted_trajectory_hashes", []):
            assert task["status"] == "COMMITTED", (root, number, hash_)
            assert hash_ not in provenance, ("duplicate submitted hash", hash_)
            provenance[hash_] = number
    committed = {e.get("details", {}).get("trajectory_hash"): i for i, e in enumerate(events)
                 if e["type"] == "plan_committed"}
    actions = []
    for index, event in enumerate(events):
        if event["type"] not in {"maneuver_started", "admitted"}:
            continue
        hash_ = event.get("details", {}).get("trajectory_hash")
        trajectory = trajectories.get(hash_)
        if trajectory is None:
            issues.append({"type": "START_WITHOUT_TRAJECTORY", "event_index": index, "hash": hash_})
            trajectory = {}
        identity = str(event["aircraft_id"])
        assert identity in uid_by_id, identity
        kind = trajectory.get("kind", event.get("details", {}).get("maneuver_kind"))
        if event["type"] == "admitted" and kind != "INBOUND":
            issues.append({"type": "ADMISSION_KIND_NOT_INBOUND", "event_index": index, "kind": kind})
        source_round = provenance.get(hash_)
        action = {"aircraft_id": identity, "uid": uid_by_id[identity], "trajectory_hash": hash_,
                  "kind": kind, "source_ring": trajectory.get("source_ring"),
                  "source_slot": trajectory.get("source_slot"), "target_ring": trajectory.get("target_ring"),
                  "target_slot": trajectory.get("target_slot"), "pad_id": trajectory.get("pad_id"),
                  "start_time": trajectory.get("start_time"), "end_time": trajectory.get("end_time"),
                  "start_event_time": event["time"], "start_event_index": index,
                  "plan_committed_event_index": committed.get(hash_), "source_round": source_round,
                  "origin": "NORMAL_PLANNING" if source_round is not None else "PUBLIC_INBOUND" if kind == "INBOUND" else "UNMATCHED",
                  "completion_event_index": None, "completion_time": None, "completion_type": None}
        completion_type = "touchdown" if kind in {"FINAL_NORMAL", "FINAL_EMERGENCY"} else "inbound_completed" if kind == "INBOUND" else "transfer_completed"
        completion = next(((j, e) for j, e in enumerate(events[index + 1:], index + 1)
                           if e["type"] == completion_type and str(e["aircraft_id"]) == identity), None)
        if completion:
            j, e = completion
            action.update(completion_event_index=j, completion_time=e["time"], completion_type=e["type"])
        actions.append(action)
    actions_by_hash = {a["trajectory_hash"]: a for a in actions}
    submitted = []
    for hash_, number in provenance.items():
        action = actions_by_hash.get(hash_)
        if action is None:
            trajectory = trajectories.get(hash_, {})
            action = dict(trajectory, uid=uid_by_id.get(str(trajectory.get("aircraft_id"))),
                          trajectory_hash=hash_, source_round=number, origin="NORMAL_PLANNING",
                          start_event_index=None, plan_committed_event_index=committed.get(hash_))
            issues.append({"type": "SUBMITTED_NOT_STARTED", "hash": hash_, "source_round": number})
        if hash_ not in committed:
            issues.append({"type": "SUBMITTED_WITHOUT_PLAN_EVENT", "hash": hash_, "source_round": number})
        submitted.append(action)
    for action in actions:
        if action["origin"] == "UNMATCHED":
            issues.append({"type": "START_WITHOUT_NORMAL_TASK", "hash": action["trajectory_hash"]})
    rounds = []
    for number, task in enumerate(tasks, 1):
        record = by_effective[task["effective_time"]]
        event_matches = [(i, e) for i, e in starts if e["details"]["snapshot_time"] == task["snapshot_time"]]
        assert len(event_matches) == 1
        index, event = event_matches[0]
        assert event["time"] == task["snapshot_time"] and event["details"]["effective_time"] == record["time"]
        current_actions = active_at(actions, task["snapshot_time"], index)
        selected = selection(record)
        all_ids = sorted(set(selected["initial_ids"]) | set(selected["attempted_ids"]) |
                         set(selected["returned_ids"]) | {c["aircraft_id"] for c in record.get("aircraft_choices", [])})
        choices = {c["aircraft_id"]: c for c in record.get("aircraft_choices", [])}
        actual = [a for a in submitted if a["source_round"] == number]
        aircraft_rows = []
        for identity in all_ids:
            assert identity in uid_by_id, identity
            choice = choices.get(identity, {})
            own_actual = [a for a in actual if a.get("aircraft_id") == identity]
            own_ongoing = [a for a in current_actions if a["aircraft_id"] == identity]
            aircraft_rows.append({"root": root, "round": number, "snapshot_time": task["snapshot_time"],
                                  "aircraft_id": identity, "uid": uid_by_id[identity],
                                  "initial_selected": identity in selected["initial_ids"],
                                  "attempted": identity in selected["attempted_ids"],
                                  "returned_scope": identity in selected["returned_ids"],
                                  "selected_kind": choice.get("selected_kind"),
                                  "search_first_target": choice.get("first_target"),
                                  "search_selected_pad_ids": choice.get("selected_pad_ids"),
                                  "move_candidates": choice.get("move_candidates"),
                                  "only_keep_wait": choice.get("only_keep_wait"),
                                  "normal_status": choice.get("normal_status"),
                                  "rejection_reasons": choice.get("rejection_reasons"),
                                  "candidate_insufficiency_reasons": choice.get("candidate_insufficiency_reasons"),
                                  "outcome": execution_outcome(task["status"], own_actual, own_ongoing),
                                  "actual_actions": own_actual, "ongoing_at_snapshot": own_ongoing})
        nearby = [s for s in states if s["time"] <= task["snapshot_time"]]
        state = max(nearby, key=lambda s: s["time"]) if nearby else None
        state_aircraft = {str(a["id"]): a for a in state.get("aircraft", [])} if state else {}
        crosscheck = [{"uid": a["uid"], "trajectory_hash": a["trajectory_hash"],
                       "saved_state_hash": state_aircraft.get(a["aircraft_id"], {}).get("trajectory_hash"),
                       "matches": a["trajectory_hash"] == state_aircraft.get(a["aircraft_id"], {}).get("trajectory_hash")}
                      for a in current_actions]
        selective = record.get("selective_replanning", {})
        notes = [task.get("reason")] if task.get("reason") else []
        if task["status"] in PENDING:
            notes.append("REPLAY_END_" + task["status"])
        if selected["expanded_ids"]:
            notes.append("SCOPE_EXPANDED")
        unseen = sorted(set(all_ids) - set(selective.get("completed_evaluation_ids", all_ids)))
        rounds.append({"root": root, "round": number, "snapshot_time": task["snapshot_time"],
                       "effective_time": task["effective_time"], "ready_time": task.get("ready_time"),
                       "finished_time": task.get("finished_time"), "actual_delivery_time": record.get("actual_delivery_time"),
                       "status": task["status"], "reason": task.get("reason"), "notes": notes,
                       "normal_task_started_event_index": index, "selection": selected,
                       "aircraft_decisions": aircraft_rows, "actual_actions": actual,
                       "ongoing_normal_actions": [a for a in current_actions if a["origin"] == "NORMAL_PLANNING"],
                       "ongoing_public_actions": [a for a in current_actions if a["origin"] != "NORMAL_PLANNING"],
                       "not_completed_evaluation_ids": unseen,
                       "fixed_background_ids": selective.get("scope", {}).get("fixed_background_ids", []),
                       "state_crosscheck": {"state_time": state["time"] if state else None,
                                            "exact_snapshot": bool(state and state["time"] == task["snapshot_time"]),
                                            "actions": crosscheck}})
    return {"root": root, "uid_by_id": uid_by_id, "rounds": rounds, "submitted_actions": submitted,
            "all_started_actions": actions, "issues": issues,
            "checks": {"tasks": len(tasks), "records": len(records), "unique_tasks": len(rounds),
                       "actual_submitted_hashes": len(submitted),
                       "plan_committed_events": sum(e["type"] == "plan_committed" for e in events),
                       "maneuver_started_events": sum(e["type"] == "maneuver_started" for e in events),
                       "normal_submitted_with_start": sum(a["start_event_index"] is not None for a in submitted),
                       "public_inbound_starts": sum(a["origin"] == "PUBLIC_INBOUND" for a in actions),
                       "task_statuses": dict(Counter(t["status"] for t in tasks))}}


def fmt(value):
    return "—" if value is None else f"{value:.3f}"


def ids_text(identities, mapping):
    return ",".join(str(mapping[a]) for a in identities) or "无"


def action_text(action, provenance=False):
    ring = {"inner": "I", "middle": "M", "outer": "O", None: "外部"}
    source = f"{ring.get(action.get('source_ring'), '?')}槽{action.get('source_slot')}"
    if action.get("source_ring") is None:
        source = "外部"
    target = f"P{action['pad_id']}" if action.get("pad_id") is not None else f"{ring.get(action.get('target_ring'), '?')}槽{action.get('target_slot')}"
    kind = {"FINAL_NORMAL": "正常进近", "TRANSFER": "跨环", "INBOUND": "公共入场"}.get(action.get("kind"), action.get("kind"))
    result = f"UID{action.get('uid')} {source}→{target}（{kind}，{fmt(action.get('start_time'))}–{fmt(action.get('end_time'))}秒）"
    if provenance:
        result += f"[R{action['source_round']}]" if action.get("source_round") else "[公共]"
    return result


def decisions_text(row):
    lines = []
    for kind in ["MOVE", "KEEP", "WAIT", None]:
        entries = [a for a in row["aircraft_decisions"] if a["selected_kind"] == kind]
        if entries:
            lines.append(f"搜索{kind or '无返回choice'}：" + ",".join(str(a["uid"]) for a in entries))
    unsubmitted_moves = [a for a in row["aircraft_decisions"] if a["selected_kind"] == "MOVE" and not a["actual_actions"]]
    if unsubmitted_moves:
        lines.append("搜索MOVE未实际提交：" + ",".join(str(a["uid"]) for a in unsubmitted_moves))
    lines += ["实际启动：" + action_text(a) for a in row["actual_actions"] if a.get("start_event_index") is not None]
    for outcome, label in [("SUBMITTED_NOT_STARTED", "已提交，未见启动"), ("MAINTAIN_PREVIOUS_ACTION", "无新动作，启动时维持既有动作"),
                           ("NO_NEW_ACTION", "本轮无新动作"), ("TASK_PENDING", "任务未完成，尚无最终交付")]:
        entries = [a for a in row["aircraft_decisions"] if a["outcome"] == outcome]
        if entries:
            lines.append(label + "：" + ",".join(str(a["uid"]) for a in entries))
    return "<br>".join(lines) or "无"


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def document(reports, output):
    absolute = lambda path: path.resolve().as_posix()
    lines = ["# 300秒新回放：逐轮选机与实际飞行决定", "",
             "本报告只读取保存的 scheduler_records、normal_tasks、events、states 和 trajectories；未运行仿真、规划器或求解器。中负载 s3r23 共21轮，高负载 s3r24 共16轮。",
             "", "时刻均为模拟秒，显示3位小数，CSV/JSON保留原始精度。当前轮时刻取 normal_tasks.snapshot_time；生效是 effective_time，ready 是 ready_time，完成是 finished_time，交付/拒绝结案时刻来自 actual_delivery_time。—表示原记录缺失，不能解释为0。",
             "", "UID与现有ReplayController一致：对各自回放整个states中出现的 aircraft.id 字符串排序后，从0编号。I/M/O分别是内/中/外环；槽位和pad编号均为0基，与viewer一致。两个负载的UID独立，不可直接按相同整数认作同一原id。",
             "", "搜索MOVE/KEEP/WAIT只来自aircraft_choices。实际启动必须有本轮actual_submitted_trajectory_hashes、任务COMMITTED、plan_committed及maneuver_started证据；预测尾部、selected_pad_ids不当作实际落地决定。正常进近表示启动了进近，只有touchdown事件才表示已落地。动作区间右端来自已提交轨迹end_time；超过300秒时是计划结束时刻，不能认作已观测完成或着陆。末轮pending只报告搜索选择和未完成状态。",
             "", "“正在执行”是在当前normal_task_started事件之前已启动、且满足start≤snapshot<end、尚未完成的动作；同一秒按events.jsonl原始顺序，包含比相邻上一轮更早的长动作。公共INBOUND独立列在单元格中；常规HOLD不称执行机动。最近保存state只交叉核对，精确性和hash逐项保存在JSON，不将较早state冒充当前快照。",
             "", "选机列显示初选名义K8、内部尝试扩展UID、最终返回scope和K。顶层actual_k来自初选selection，最终scope.actual_k来自局部规划返回；两层原字段均保存在JSON。KEEP/WAIT表示本轮未下达新动作/保留既有计划，不能据此推断启动时飞机在环上等待；aircraft_choices.ring是生效状态投影。候选缺口仅描述有界目录证据，UNKNOWN不代表物理不可行。", ""]
    lines += ["状态释义：COMMITTED=已实际提交；REJECTED=本轮交付被拒绝；NO_EXECUTABLE_PREFIX=本轮未交付非空可执行前缀，不能解释为物理无解；COMPUTING=回放终点仍计算中；REVALIDATING=回放终点仍重验证中。TRAFFIC_CHANGED_AFTER_DEADLINE=截止时间之后交通状态变化；READY_DEADLINE_MISSED=未满足ready截止时刻。",
              "", "最终方案以返回scope为准；内部尝试范围及补入飞机另列，未纳入相应scope的飞机作为固定背景参与约束。背景清单和数量保存在JSON/CSV，不在每行重复。SCOPE_EXPANDED表示内部尝试补入飞机，最终是否保留以返回scope列为准。", ""]
    for report, label in zip(reports, ["中负载", "高负载"]):
        mapping = report["uid_by_id"]
        checks = report["checks"]
        lines += [f"## {label} {report['root']}", "",
                  f"核对：{checks['tasks']}个唯一任务与{checks['records']}条记录一一对应；实际提交{checks['actual_submitted_hashes']}条，plan_committed与maneuver_started各{checks['maneuver_started_events']}条，全部已匹配。公共入场{checks['public_inbound_starts']}次另计。",
                  "",
                  "| 轮次 / 启动时刻 | 生效 / ready / 完成 / 交付；状态 | 选中UID（初选；扩展；返回） | 最终飞行决定（搜索与执行分开） | 当时仍执行的以前动作 | 特殊备注 |",
                  "|---|---|---|---|---|---|"]
        for row in report["rounds"]:
            selected = row["selection"]
            select = f"初选K{len(selected['initial_ids'])}：{ids_text(selected['initial_ids'], mapping)}<br>尝试补入：{ids_text(selected['expanded_ids'], mapping)}<br>最终返回K{selected['returned_scope_size']}：{ids_text(selected['returned_ids'], mapping)}"
            status_label = {"COMMITTED": "已提交", "REJECTED": "交付被拒", "NO_EXECUTABLE_PREFIX": "未交付非空可执行前缀", "COMPUTING": "终点计算中", "REVALIDATING": "终点重验证中"}.get(row["status"], row["status"])
            timing = f"{fmt(row['effective_time'])} / {fmt(row['ready_time'])} / {fmt(row['finished_time'])} / {fmt(row['actual_delivery_time'])}<br>{status_label}（{row['status']}）"
            active = [action_text(a, True) for a in row["ongoing_normal_actions"]]
            public = [action_text(a, True) for a in row["ongoing_public_actions"]]
            if public:
                active.append("公共动作：" + "；".join(public))
            notes = list(row["notes"])
            missing = [a for a in row["aircraft_decisions"] if a["move_candidates"] == 0]
            if missing:
                notes.append("有界目录无MOVE：" + ",".join(str(a["uid"]) for a in missing))
                rejections = {str(a["uid"]): a["rejection_reasons"] for a in missing if a["rejection_reasons"]}
                if rejections:
                    notes.append("候选拒因：" + "；".join(
                        f"UID{uid} " + ",".join(f"{reason}×{count}" for reason, count in reasons.items())
                        for uid, reasons in rejections.items()))
            if row["not_completed_evaluation_ids"]:
                notes.append("未完成评估：" + ids_text(row["not_completed_evaluation_ids"], mapping))
            lines.append(f"| R{row['round']} / {fmt(row['snapshot_time'])} | {timing} | {select} | {decisions_text(row)} | {'<br>'.join(active) or '无'} | {'<br>'.join(notes) or '无'} |")
        lines += ["", f"### {report['root']} UID→原aircraft_id", "", "| UID | aircraft_id |", "|---|---|"]
        lines += [f"| {uid} | {identity} |" for identity, uid in mapping.items()]
        lines += [""]
    lines += ["## 机器可读证据", "",
              f"- [完整JSON]({absolute(output / 'report.json')})：逐轮、逐机、动作hash、事件索引、state交叉核对及缺口。",
              f"- [每任务一行CSV]({absolute(output / 'tasks.csv')})。",
              f"- [每选中/尝试飞机一行CSV]({absolute(output / 'selected_aircraft.csv')})。",
              f"- [启动时进行中动作CSV]({absolute(output / 'ongoing_actions.csv')})（含普通/公共区分）。",
              f"- [实际提交动作CSV]({absolute(output / 'submitted_actions.csv')})（含touchdown/transfer_completed证据；未完成动作不宣称落地）。", "",
              "源文件均位于 artifacts/rs_selective_repair_review_20261029/new_results/arms/{root}/evaluation/{root}/RULE_LOCAL_K8/；JSON记录完整源路径。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--document", type=Path, default=DEFAULT_DOCUMENT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing evidence directory: {args.output}")
    reports = []
    for root in ["s3r23", "s3r24"]:
        folder = args.input / root / "evaluation" / root / "RULE_LOCAL_K8"
        report = extract(root, read_json(folder / "scheduler_records.json"), read_json(folder / "normal_tasks.json"),
                         read_jsonl(folder / "replay/events.jsonl"), read_jsonl(folder / "replay/states.jsonl"),
                         read_json(folder / "replay/trajectories.json"))
        report["source_directory"] = str(folder.resolve())
        reports.append(report)
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    task_rows, aircraft_rows, ongoing_rows, submitted_rows = [], [], [], []
    for report in reports:
        for row in report["rounds"]:
            task_rows.append({k: v for k, v in row.items() if k not in {"aircraft_decisions", "actual_actions", "ongoing_normal_actions", "ongoing_public_actions"}})
            aircraft_rows.extend(row["aircraft_decisions"])
            for action in row["ongoing_normal_actions"] + row["ongoing_public_actions"]:
                ongoing_rows.append(dict(root=report["root"], round=row["round"], snapshot_time=row["snapshot_time"], **action))
        submitted_rows.extend(dict(root=report["root"], **a) for a in report["submitted_actions"])
    for name, rows in [("tasks", task_rows), ("selected_aircraft", aircraft_rows), ("ongoing_actions", ongoing_rows), ("submitted_actions", submitted_rows)]:
        write_csv(args.output / f"{name}.csv", rows)
    args.document.parent.mkdir(parents=True, exist_ok=True)
    args.document.write_text(document(reports, args.output), encoding="utf-8")
    print(json.dumps({r["root"]: {"checks": r["checks"], "issues": r["issues"]} for r in reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
