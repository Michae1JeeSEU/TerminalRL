低空交通自组织与宏微观协同资料包
================================

核对日期：2026-10-13

推荐题目
--------

面向起降瓶颈的低空交通自组织机理与宏微观协同调控方法研究

英文题目：Self Organization and Macro Micro Coordinated Control of Low Altitude Traffic with Vertiport Bottlenecks。

研究计划和论文框架见 02、03 文件。题目、研究边界和贡献表述以 02_research_proposal.txt 为准；本 README 只说明资料入口、文件状态和可核验计数。

文件入口
--------

- [01_literature_synthesis.txt](01_literature_synthesis.txt) / [01_literature_synthesis.pdf](01_literature_synthesis.pdf)：文献综述、证据边界和选题判断；TXT 与 PDF 均已存在。
- [02_research_proposal.txt](02_research_proposal.txt) / [02_research_proposal.pdf](02_research_proposal.pdf)：research proposal；TXT 与 PDF 均已存在。
- [03_dissertation_outline.txt](03_dissertation_outline.txt) / [03_dissertation_outline.pdf](03_dissertation_outline.pdf)：博士论文框架；TXT 与 PDF 均已存在。
- [04_literature_review.xlsx](04_literature_review.xlsx)：文献矩阵表；文件已存在，可从此处进入。
- [05_fulltext_to_download.txt](05_fulltext_to_download.txt)：全文获取清单。I 类是尚未阅读全文的条目，II 类单列已读 HTML 但没有本地文件的 A08。
- [evidence/artifact_inventory.json](evidence/artifact_inventory.json)：33 条记录的 ID、DOI、来源入口、阅读状态、本地路径、文件大小、PDF 签名、解析状态和页数观察。
- [evidence/track_A.json](evidence/track_A.json)、[evidence/track_B.json](evidence/track_B.json)、[evidence/track_C.json](evidence/track_C.json)：三条文献检索轨迹的原始证据表。
- `originals/`：成功取得的本地 PDF；`extracted/`：相应文本提取结果。

实际计数
--------

- 文献记录共 33 条：A 10 条、B 12 条、C 11 条；ID 唯一，全部有 DOI 和 primary URL。
- 按 track 中的 `reading_status`：FULL_TEXT 19 条，ABSTRACT_ONLY 14 条。A08 的公开 HTML 正文已读，虽然没有本地文件，仍计入 FULL_TEXT，不放入“未阅读全文”清单。
- 18 条文献记录有本地主 PDF 和 TXT。A08 没有本地 PDF/TXT；A09 的主文材料是 `originals/A09_article.pdf` 与 `extracted/A09_article.txt`。
- `originals/` 和 `extracted/` 子目录中实际观察到 19 个 PDF 和 19 个 TXT：额外的一对 `A09.pdf`、`A09.txt` 是 A09 相关学位论文补充材料，不另计为一条文献记录。
- 19 个实际 PDF 均具有 `%PDF-` 文件签名并可由 pypdf 解析；页数和路径见 `evidence/artifact_inventory.json`。
- 14 条 ABSTRACT_ONLY 的获取顺序和缺失原因见 05 文件；其中 B05、B12、B06、C04、A06、B04、B11、C02、B03 按任务指定顺序排在前面。

版本和阅读状态
--------------

三条 track 的 `version` 计数为 publisher 26、accepted 3、preprint 4。`publisher` 表示出版商页面或正式出版版本；`accepted` 表示作者接受稿；`preprint` 表示用于阅读的预印本，可能与最终排版版在页码、图表或文字上不同。引用时以 DOI、正式期刊/会议和正式年份为准；页码级证据应同时注明所读本地版本。

A09 的 43 页 `A09_article.pdf` 是主文阅读材料，244 页 `A09.pdf` 是题为 “Essays on deep reinforcement learning with applications to path planning and control of autonomous vehicles” 的学位论文补充材料。A08 是已读公开 HTML 正文，但没有本地 PDF/TXT，不能把“没有本地文件”误写成“没有阅读全文”。

版权与本地文件
--------------

部分原文本次未能通过公开来源获取，或只能获得作者稿、预印本、机构页面和摘要；这些状态已经在 05 清单及各 track 的 `access_attempts` 中注明。`originals/` 中的原始版权文件只保留在本地工作目录，不上传 Git；`.gitignore` 仅排除原始全文、提取文本、QA 和依赖缓存等支持文件，不排除本 README、综述、proposal、论文框架、文献矩阵、脚本或 evidence JSON。
