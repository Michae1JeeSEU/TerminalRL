from pathlib import Path
p=Path('docs/CURRENT_STATUS.md');s=p.read_text(encoding='utf8')
marker='## 2026-10-03 P3 统一实现修复（最新）'
line='- 新fit_initial已FIT_FROZEN/exit0：91901–3各fresh5000updates完成，严格1298新train行；canonical91901 SHA16d3ee800263023386341cb92a6335bb63dd487a346181a0275bf4c3ecb63600，训练内18条R→W残差保留，不挑seed、不据拟合授资格。新student smoke892864通过三接口3R与2tick重放；flow892853完整667ticks/6N/0E/F/D/U、98actor/18R/0shield，实际安全/hash无异常，全请求等待mean14.893s，完整重放与逐行分析通过。primary接受下一唯一dense24_fixed892854/929854，仍为development；无新Q/critic/PPO/formal，详见[新轮记录](evidence/v4_2/p3_v25_nominal_curve_fresh_training_20261008.md)。\n\n'
assert s.count(marker)==1 and line not in s
p.write_text(s.replace(marker,line+marker),encoding='utf8')
b=Path('.tmp/commit_p3_fresh_data_complete.py').read_text(encoding='utf8')
b=b.replace("startswith('4a4e170')","startswith('56ad3cb')").replace('docs: accept fresh P3 course data for fixed imitation fitting','docs: record fresh P3 fit and complete flow student validation')
Path('.tmp/commit_p3_fresh_student_flow.py').write_text(b,encoding='utf8')
