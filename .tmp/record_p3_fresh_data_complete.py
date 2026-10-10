from pathlib import Path
p=Path('docs/CURRENT_STATUS.md')
s=p.read_text(encoding='utf8')
marker='## 2026-10-03 P3 统一实现修复（最新）'
line='- 修复合同下三组新train已全部COLLECTION_OK：flow5N1E、fixed20N4E、mixed14N10E，共54请求39N15E/0F/D/U，无实际安全/hash/技术/censor阻塞。primary核对24868raw/1298有效标签（204R1094W、19种proposal），forced与null未入actor loss；随机任务等级仍关闭，保留总等待/SOC紧迫度与物理服务。mixed标注4546.196s、最低完成SOC .126563，三组全部在预算内，终态文件两端一致、原进程已退出。已准许既定fresh91901–3固定5000updates拟合，新fit后继续完整student开发闭环；尚无新学生/Q/PPO/formal结果，不能宣称训练退化解决。详见[新数据与训练记录](evidence/v4_2/p3_v25_nominal_curve_fresh_training_20261008.md)。\n\n'
assert s.count(marker)==1 and line not in s
p.write_text(s.replace(marker,line+marker),encoding='utf8')
b=Path('.tmp/commit_p3_fresh_dense.py').read_text(encoding='utf8')
b=b.replace("startswith('2c4384c')","startswith('4a4e170')").replace('docs: accept complete dense fixed P3 collection under repaired contract','docs: accept fresh P3 course data for fixed imitation fitting')
Path('.tmp/commit_p3_fresh_data_complete.py').write_text(b,encoding='utf8')
