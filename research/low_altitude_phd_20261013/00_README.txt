低空交通自组织与宏微观协同引导资料包
==================================

核对日期：2026-10-22

推荐题目
--------
低空交通自组织运行机理与宏微观协同引导方法研究

研究独立OD的低空多向交通，从单交汇区扩展到3—4个相邻区域；在固定总空间和安全标准下，研究需求强度/流向组成改变时自组织是否足够、何时及多强引导、如何与准入/边界额度协调。引导采用软方向场或建议通行带，个体保留局部速度/短轨迹决策；不涉及终端设施或SOC主线。研究对象由文献及用户兴趣推导。

文件入口
--------
- 01_literature_synthesis.txt / .pdf：文献综述、已有成果边界与新选题判断。
- 02_research_proposal.txt / .pdf：研究问题、方法、对照及范围。
- 03_dissertation_outline.txt / .pdf：博士论文框架与章节关系。
- 04_literature_review.xlsx：35篇文献矩阵。
- 05_fulltext_to_download.txt：剩余获取状态，A06/C03按用户要求停止追索；补读与新增原文映射。
- 06_revision_notes_20261022.txt：脱离现有项目的选题比较与修订说明。
- evidence/track_A.json、track_B.json、track_C.json：元数据、方法、结果、局限和获取状态；旧失败记录保留为历史，当前状态以更新字段为准。
- evidence/supplement_assignment.json、supplement_macro_control.json、supplement_structure_scheduling.json：12篇补充全文的页码与旧判断纠正。
- evidence/primary_critique.json：逐篇研究评述与相对新题目的关联，包含外围参考。
- evidence/artifact_inventory.json：当前track与真实原文文件的路径、大小、页数和计数核对。
- tools/：生成与检查脚本；originals/、supplement/、extracted/、qa/：本地原文、补充原文、文本与视觉核对支持文件。

实际计数
--------
35条文献：A类12条、B类12条、C类11条。33条FULL_TEXT、2条ABSTRACT_ONLY（A06、C03）。31条有本地主文PDF；另外两篇已读远程全文分别是A08的publisher HTML、A11的完整remote PDF，不能统称HTML或据无本地文件认定未读。

originals/有20个PDF，supplement/有12个PDF，共32个：31篇主文及A09的1篇补充学位论文。A11本地下载未成功，local_pdf/local_text为null，不虚构路径。A12为本地8页正式会议PDF。12篇用户补充ID是A03、B02、B03、B04、B05、B06、B07、B11、B12、C02、C04、C06，均已阅读全文。

A09主文是originals/A09_article.pdf（43页），originals/A09.pdf（244页）是相关学位论文，不另计文献。B07是6页会议全文，已完整视觉阅读；extracted/supplement_B07.txt含可提取页眉与明确标记的人工笔记，不是逐字OCR。

版本与使用边界
--------------
以version、DOI及实际所读版本确认页码；作者稿与正式版可能不同。A11通过公开作者PDF完整阅读，A12未核到DOI故留空。A06/C03只按摘要限界引用并停止追索，不要求补齐后才推进。FULL_TEXT是阅读媒介与覆盖状态，不自动代表本地下载成功。

现有研究已经覆盖方向规则、参数优化、策略相关容量、路线引导和宏微反馈。候选贡献是同资源下机制分离与条件供给迁移、需求改变后的组织过渡，以及计入两者的联合控制增量，均待实验验证。规则及小规模优化先行，强化学习和硬件可选。本轮没有启动新实验或产生方法性能结果。

版权与本地文件
--------------
originals/、supplement/中的原文只作本地资料，不纳入本轮Git提交。资料包.gitignore排除两原文目录、extracted/、qa/及依赖缓存，保留报告、文献矩阵、生成脚本和evidence JSON。PDF与表格须重新生成并核验；文件存在本身不代替内容核验。
