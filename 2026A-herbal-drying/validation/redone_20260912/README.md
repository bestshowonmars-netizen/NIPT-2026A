# 第三、四问重算档案（2026-09-12）

第三问从现有第二问10800 s完整状态重新积分，第四问从题设初态重新积分。
严格全域判据为 \(\max_r C(r,t)<0.15\)，得到总时间57.4723 h与51.0871 h，均为中心最后达标。
两问保存的全部数值数组（包括无效位置掩码）与原验收解完全一致。
本轮补充了收缩坐标与守恒推导，修订了精度说明、方法短稿及图5—7。

阅读[验证说明](验证说明.md)、[第三问解答](../../第3问解答.md)和[第四问解答](../../第4问解答.md)。
正式结果位于项目的 `submission_results`，图表位于 `q3`、`q4`。

## 复现入口

以下命令均在 `2026A-herbal-drying` 项目根目录执行，并先按项目README安装依赖。

```powershell
# 使用已保存的正式状态，检查结果并重新导出。
python -B run_all.py --question 3 --validate
python -B run_all.py --question 4 --validate

# 强制重新积分；耗时取决于CPU。
python -B run_all.py --question 3 --recompute --compute-only
python -B run_all.py --question 4 --recompute --compute-only

# 用本目录保存的本轮状态，核验后生成独立解答、图表和压缩包。
# 输出置于被Git忽略的.runtime目录，不覆盖正式结果。
python -B validation/rebuild_q3_q4_results.py --output .runtime/redone_delivery
```

最后一个入口会核对两问的新状态、完成记录、原始加密证据和逐格Excel内容。
它复用本轮已经保存的状态，不再次运行长时间积分。添加 `--publish-project` 才会同步
更新项目中的解答、图表和Excel；原数值档案保持保留。

## 档案内容与边界

- `q3/solution.npz`、`q4/solution.npz`：本轮未舍入结果；同名JSON为模型与运行元数据。
- 各问 `launch.json`、`receipt.json`：输入、配置、完成时间、文件哈希及核查结果。
- `recomputed.json`：数组比较、来源绑定、加密证据重核及导出检查。
- `q4/preflight.json`：固定半径退化、无交换均匀场、重启及输出位置核查，并复核原固定半径对照。
- `journal_verification.json`：对本轮首个分钟区间的缓存回放核查，温度、含水率及诊断量均与直接积分一致。

原始记录可能带有计算机器的绝对路径，作为历史来源保留；重建入口用项目内相对位置
解析这些记录，并核验内容哈希，不要求复现者具有相同盘符和用户名。

`q3/recompute.py`、`q4/run.py`、`q4/verify.py` 与上级的 `recompute_durable.py`
保留本轮实际执行方式。它们是单次运行驱动：第四问会拒绝覆盖已存在的证据，
第三问的驱动可能重写其同目录记录，勿将它们当作默认复现命令。
`check_interval_journal.py` 需要该轮的 `.runtime` 分钟区间缓存；这些缓存不推送，
因此干净克隆应使用上面的统一入口，不能将缺少缓存导致的回放失败当作求解失败。
`finalize_redone.py` 只是等待两问完成后导出的本地辅助程序。

中断日志、预览文件、临时重启状态和依赖缓存不属于此归档。第四问固定半径对照及
整个加密矩阵未在本轮重新积分，本轮重新核对的是此前保存的原始证据。
重算一致性证明可复现性，数值核查不替代模型假设及内部温湿场的实验验证。
