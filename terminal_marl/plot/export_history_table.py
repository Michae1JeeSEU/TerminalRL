from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


# ========= 需要修改的参数 =========
HISTORY_PATH = Path(
    r"D:/qidong/毕设/代码/MDP+PPO/260411/2/terminal_marl_env/outputs/shared_ppo_run/history.jsonl"
)
AIRCRAFT_COUNT = 6
# =================================


def load_history_jsonl(history_path: Path) -> pd.DataFrame:
    if not history_path.exists():
        raise FileNotFoundError(f"找不到 history.jsonl 文件: {history_path}")

    records = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {line_idx} 行 JSON 解析失败: {e}") from e

    if not records:
        raise ValueError("history.jsonl 为空，无法导出表格。")

    return pd.DataFrame(records)


def reorder_columns(df: pd.DataFrame, preferred_cols: list[str]) -> pd.DataFrame:
    existing_cols = [c for c in preferred_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_cols]
    return df[existing_cols + other_cols]


def build_train_summary(df: pd.DataFrame, aircraft_count: int) -> pd.DataFrame:
    train_cols = [
        "update",
        "agent_steps",
        "collected_agent_steps",
        "completed_env_episodes",
        "mean_agent_return",
        "mean_agent_ep_len",
        "episode_time",
        "episode_completion_time",
        "episode_step",
        "episode_completion_step",
        "episode_finished",
        "episode_sequenced",
        "episode_mean_completion_time_finished",
        "episode_max_completion_time_finished",
        "episode_mean_path_length",
        "episode_mean_path_detour_rate",
        "episode_mean_waiting_ring_distance",
        "episode_mean_cumulative_turn_angle_deg",
        "episode_mean_cumulative_height_change",
        "episode_mean_residual_action_l1",
        "episode_collision_count",
        "episode_nmac_count",
        "lr",
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_frac",
        "elapsed_sec",
    ]

    train_cols = [c for c in train_cols if c in df.columns]
    train_df = df[train_cols].copy()

    if "episode_finished" in train_df.columns:
        train_df["episode_success_rate"] = train_df["episode_finished"] / float(aircraft_count)

    preferred_order = [
        "update",
        "agent_steps",
        "collected_agent_steps",
        "completed_env_episodes",
        "mean_agent_return",
        "mean_agent_ep_len",
        "episode_finished",
        "episode_success_rate",
        "episode_mean_completion_time_finished",
        "episode_mean_path_length",
        "episode_mean_path_detour_rate",
        "episode_collision_count",
        "episode_nmac_count",
        "lr",
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_frac",
        "elapsed_sec",
    ]
    return reorder_columns(train_df, preferred_order)


def build_eval_summary(df: pd.DataFrame, aircraft_count: int) -> pd.DataFrame:
    eval_cols = [
        "update",
        "eval_mean_agent_return",
        "eval_time",
        "eval_completion_time",
        "eval_step",
        "eval_completion_step",
        "eval_finished",
        "eval_sequenced",
        "eval_mean_completion_time_finished",
        "eval_max_completion_time_finished",
        "eval_mean_path_length",
        "eval_mean_path_detour_rate",
        "eval_mean_waiting_ring_distance",
        "eval_mean_cumulative_turn_angle_deg",
        "eval_mean_cumulative_height_change",
        "eval_mean_residual_action_l1",
        "eval_collision_count",
        "eval_nmac_count",
    ]

    eval_cols = [c for c in eval_cols if c in df.columns]
    if not eval_cols:
        return pd.DataFrame()

    eval_df = df[eval_cols].copy()

    # 只保留真正有评估数据的行
    non_update_cols = [c for c in eval_df.columns if c != "update"]
    if non_update_cols:
        eval_df = eval_df.dropna(how="all", subset=non_update_cols)

    if eval_df.empty:
        return eval_df

    if "eval_finished" in eval_df.columns:
        eval_df["eval_success_rate"] = eval_df["eval_finished"] / float(aircraft_count)

    preferred_order = [
        "update",
        "eval_mean_agent_return",
        "eval_finished",
        "eval_success_rate",
        "eval_mean_completion_time_finished",
        "eval_mean_path_length",
        "eval_mean_path_detour_rate",
        "eval_collision_count",
        "eval_nmac_count",
    ]
    return reorder_columns(eval_df, preferred_order)


def main() -> None:
    df = load_history_jsonl(HISTORY_PATH)

    out_dir = HISTORY_PATH.parent
    full_csv_path = out_dir / "history_full_table.csv"
    train_csv_path = out_dir / "history_train_summary.csv"
    eval_csv_path = out_dir / "history_eval_summary.csv"

    # 完整表格：常用字段排前面，其余字段保留
    full_preferred_cols = [
        "update",
        "agent_steps",
        "collected_agent_steps",
        "completed_env_episodes",
        "mean_agent_return",
        "mean_agent_ep_len",
        "episode_finished",
        "episode_mean_completion_time_finished",
        "episode_mean_path_length",
        "episode_mean_path_detour_rate",
        "episode_collision_count",
        "episode_nmac_count",
        "eval_mean_agent_return",
        "eval_finished",
        "eval_mean_completion_time_finished",
        "eval_mean_path_length",
        "eval_mean_path_detour_rate",
        "eval_collision_count",
        "eval_nmac_count",
        "lr",
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_frac",
        "elapsed_sec",
    ]
    full_df = reorder_columns(df.copy(), full_preferred_cols)

    if "episode_finished" in full_df.columns:
        full_df["episode_success_rate"] = full_df["episode_finished"] / float(AIRCRAFT_COUNT)
    if "eval_finished" in full_df.columns:
        full_df["eval_success_rate"] = full_df["eval_finished"] / float(AIRCRAFT_COUNT)

    # 导出完整表格
    full_df.to_csv(full_csv_path, index=False, encoding="utf-8-sig")

    # 导出训练摘要表
    train_df = build_train_summary(df, AIRCRAFT_COUNT)
    train_df.to_csv(train_csv_path, index=False, encoding="utf-8-sig")

    # 导出评估摘要表
    eval_df = build_eval_summary(df, AIRCRAFT_COUNT)
    if not eval_df.empty:
        eval_df.to_csv(eval_csv_path, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print(f"已读取日志文件: {HISTORY_PATH}")
    print(f"总记录数: {len(df)}")
    print(f"总字段数: {len(df.columns)}")
    print("字段名如下：")
    print(df.columns.tolist())
    print("=" * 60)
    print(f"已导出完整表格: {full_csv_path}")
    print(f"已导出训练摘要表: {train_csv_path}")
    if not eval_df.empty:
        print(f"已导出评估摘要表: {eval_csv_path}")
    else:
        print("未检测到有效评估字段，未导出评估摘要表。")
    print("=" * 60)

    print("\n训练摘要表前 5 行：")
    print(train_df.head().to_string(index=False))

    if not eval_df.empty:
        print("\n评估摘要表前 5 行：")
        print(eval_df.head().to_string(index=False))


if __name__ == "__main__":
    main()