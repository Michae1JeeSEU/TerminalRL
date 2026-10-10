import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 1. 读取数据（只读取需要的2列，避免多余数据干扰） ----------------------
df = pd.read_csv("D:/qidong/毕设/代码/MDP+PPO/260411/2/terminal_marl_env/outputs/shared_ppo_run/history_full_table.csv", usecols=["update", "mean_agent_return"])
x = df["update"]
y = df["mean_agent_return"]

# 滑动窗口设置（和你当前平滑效果一致）
window_size = 15
# 计算滑动平滑+局部标准差
y_smooth = y.rolling(window=window_size, center=True, min_periods=1).mean()
y_std = y.rolling(window=window_size, center=True, min_periods=1).std()
# 置信区间上下界（仅用于填充，绝对不绘制线条）
upper = y_smooth + y_std
lower = y_smooth - y_std

# ---------------------- 2. 全局样式设置 ----------------------
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(10, 5), dpi=150)

# ---------------------- 3. 核心绘图（彻底杜绝上下线） ----------------------
# 仅绘制1条主平滑曲线，绝对不绘制上下界的线条
plt.plot(x, y_smooth, color="#1F77B4", linewidth=0.2, label="Mean Reward")

# 填充置信区间阴影，强制关闭边缘线（这是解决问题的核心！）
plt.fill_between(
    x, lower, upper,
    color="#1F77B4",
    alpha=0.3,
    edgecolor="none",  # 彻底关闭阴影的上下边缘线
    linewidth=0         # 强制边缘线宽度为0，双重保险
)

# ---------------------- 4. 样式和参考图完全对齐 ----------------------
plt.grid(True, linestyle="--", alpha=0.3, zorder=0)
plt.xlabel("Update", fontsize=12)
plt.ylabel("Mean Agent Return", fontsize=12)
plt.title("Training Reward Curve", fontsize=14, pad=12)
plt.legend(loc="lower right", frameon=True, edgecolor="#333333", facecolor="white", fontsize=10)
plt.tick_params(axis='both', labelsize=10)
plt.tight_layout()

# 保存高清图
plt.savefig("reward_curve_final_no_bound.png", dpi=300, bbox_inches="tight")
plt.show()