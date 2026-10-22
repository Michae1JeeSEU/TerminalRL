低空终端交通自组织与宏微观协同资料包
====================================

核对日期：2026-10-22

推荐题目
--------
能量约束下低空终端交通自组织与宏微观协同调度研究

当前范围：单起降场、固定构形、同质UAV、一个pad、2—4股进场流和随机入场SOC。固定多环是可选主实例；在线空域重构列为可选扩展。题目、研究问题和方法详见02；本README说明资料入口与可核验文件状态。

文件入口
--------
- 01_literature_synthesis.txt / 01_literature_synthesis.pdf：更新后的文献综述、原文边界与本次选题判断。
- 02_research_proposal.txt / 02_research_proposal.pdf：研究计划、问题、方法、对照与范围。
- 03_dissertation_outline.txt / 03_dissertation_outline.pdf：博士论文框架与章节关系。
- 04_literature_review.xlsx：33篇文献矩阵。
- 05_fulltext_to_download.txt：剩余全文获取状态，按用户要求停止追索；附本次12篇补充全文映射。
- 06_revision_notes_20261022.txt：本次全文补读造成的判断修订、范围收敛及未解决证据边界。
- evidence/track_A.json、track_B.json、track_C.json：文献编号、元数据、阅读状态和内容证据；原先获取失败的记录保留为历史，当前状态以补读后的字段为准。
- evidence/supplement_assignment.json、supplement_macro_control.json、supplement_structure_scheduling.json：本次12篇全文阅读的细节、页码及旧判断纠正。
- evidence/primary_critique.json：主对话研究评述。
- evidence/artifact_inventory.json：按当前track和真实文件核对的路径、大小、页数及计数。
- tools/：现有资料包生成与检查脚本；originals/、supplement/、extracted/、qa/：本地原文、补充原文、提取文本与视觉核对支持文件。

实际计数
--------
文献共33条：A10条、B12条、C11条。31条FULL_TEXT，其中30条有本地主文PDF，A08已读公开HTML且没有本地全文文件；2条ABSTRACT_ONLY为A06、C03。A08不属于第三篇未读文献。

originals/保留19个PDF，覆盖18篇主文及A09的1篇补充学位论文；supplement/为用户补充的12个PDF。两目录合计31个PDF，覆盖30篇主文及1篇学位论文。12篇补读ID为A03、B02、B03、B04、B05、B06、B07、B11、B12、C02、C04、C06，均已阅读全文，不再列为待下载。

A09主文是originals/A09_article.pdf（43页），originals/A09.pdf（244页）为相关学位论文，不另计一篇主文。B07为6页会议全文，正文无法直接提取；已渲染全部页视觉阅读，extracted/supplement_B07.txt包含可提取页眉及明确标记的人工阅读笔记，不是逐字OCR全文。

版本与使用边界
--------------
以各条version、DOI和本地PDF共同确认阅读版本；作者稿、预印本与正式出版版本可能页码不同，页码证据必须对应实际所读文件。补充PDF已核对题名和DOI。只有摘要的A06、C03仍按摘要限界引用，不补写未经核验的作者局限；缺文不阻碍当前建议推进，也不继续追索。

自组织是否成立取决于局部决策权：飞行器决定候选意图，宏观层发公共资源/流量信号并可进行占用仲裁；中心逐机选择完整动作的算法作为集中式对照。现有固定环与SOC调度代码可以支持实验，不自动等于已有自组织实现。

版权与本地文件
--------------
originals/和supplement/中的用户原文仅作为本地资料，不纳入本轮Git提交。资料包.gitignore同时排除上述两目录、extracted/、qa/和依赖缓存，保留报告、文献矩阵、生成脚本与evidence JSON。PDF/表格的再生成与核验由本轮相应构建步骤完成，文件存在本身不代替内容核验。
