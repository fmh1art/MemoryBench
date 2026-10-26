# Eval Formulation

GT 是 reference。仅评价标注的 memory targets；Environment 默认关闭且不参与 Memory Score。

## Sampling

在 GT 与 Gen 的共享时间轴上，对中间 80% 均匀采样 8 帧，包含两端：

$$
t_{i,k}=T_i\left(0.1+\frac{0.8(k-1)}{7}\right),\qquad k=1,\ldots,8.
$$

每次 Memory Judge 输入同一时间点的两张独立图像。Gate 保持一次 query，输入分别包含 12 帧的 GT/Gen 拼图。

## Gate

$$
g_i^{\mathrm{obs}},g_i^{\mathrm{act}}\in\{0,1\},\qquad
 g_i=g_i^{\mathrm{obs}}g_i^{\mathrm{act}}.
$$

任一 Gate 失败则停止 Memory Judge，该 case 总分为 0，子指标未计算。

## Inventory

VLM 标注语义组及数量；Python 计算。以下公式仅用于数量可可靠标注的帧：

$$
G_{i,k}=\sum_j n^{\mathrm{GT}}_{i,k,j},\qquad
P_{i,k}=\sum_j n^{\mathrm{Gen}}_{i,k,j}.
$$

$$
\mathrm{TP}_{i,k}=\sum_j\min(n^{\mathrm{GT}}_{i,k,j},n^{\mathrm{Gen}}_{i,k,j}),\qquad
\mathrm{FN}_{i,k}=G_{i,k}-\mathrm{TP}_{i,k},\qquad
\mathrm{FP}_{i,k}=P_{i,k}-\mathrm{TP}_{i,k}.
$$

$$
S^{\mathrm{presence}}_{i,k}=\frac{\mathrm{TP}_{i,k}}{G_{i,k}},\qquad
S^{\mathrm{count}}_{i,k}=1-\frac{|P_{i,k}-G_{i,k}|}{\max(P_{i,k},G_{i,k},1)}.
$$

$$
S^{\mathrm{Inv}}_{i,k}=0.7S^{\mathrm{presence}}_{i,k}+0.3S^{\mathrm{count}}_{i,k}.
$$

$$
R^{\mathrm{missing}}_{i,k}=\frac{\mathrm{FN}_{i,k}}{G_{i,k}},\qquad
R^{\mathrm{spurious}}_{i,k}=\frac{\mathrm{FP}_{i,k}}{\max(P_{i,k},1)}.
$$

Missing/Spurious 仅为诊断。GT 数量为 0 时，Presence 和 Missing Rate 不适用，仅使用 Count。
GT 全部目标数量明确、Gen 对应区域已展示，但生成缺陷使数量不可辨认时，适用的 Presence/Count 计 0；未知计数和 TP/FN/FP 等诊断保留 null，不能伪造 Gen count=0。

## Appearance and Spatial

VLM 给出 0–5 的整数评分，归一化后使用固定权重：

$$
S^{\mathrm{App}}_{i,k}=0.5\frac{r^{\mathrm{visual}}_{i,k}}{5}+0.5\frac{r^{\mathrm{shape}}_{i,k}}{5}.
$$

$$
S^{\mathrm{Spa}}_{i,k}=0.4\frac{r^{\mathrm{loc}}_{i,k}}{5}+0.6\frac{r^{\mathrm{rel}}_{i,k}}{5}.
$$

Appearance 在帧级仅评价匹配物体；没有匹配时帧分数为 null。若整个 case 都因 GT 明确存在的目标未恢复而无法匹配，则 case 级 Appearance 为 0。
Spatial 允许合理视角差异，优先比较稳定锚点和物体关系。

## Frame-level null and zero

子指标分数统一写为：

$$
s_{i,k,m}=\begin{cases}
\widehat{s}_{i,k,m},&\text{可可靠评价},\\
0,&\text{GT 依据明确，Gen 恢复失败使该事实不可辨认},\\
\mathrm{null},&\text{GT 未展示、GT 信息不足、属性不适用或合理遮挡},\\
\mathrm{null},&\text{Appearance 无匹配物体（仅帧级）}.
\end{cases}
$$

其中，Inventory 的正常分数由数量公式计算，Appearance/Spatial 的正常分数为 VLM 原始分数除以 5。
GT 是否展示可评价事实决定参考依据；不能仅因 Gen 生成失败而跳过该帧。自然遮挡不能自动视为生成缺陷。整个 Read 阶段未按动作/相机要求访问目标区域，属于 Interaction Physics 失败。

null 必须记录原因：reference_unavailable、natural_occlusion、generated_failure、no_matched_objects、not_applicable 或 unknown。API/解析失败属于调用错误，需要重试，不属于分数 0。

有效子指标集合包含真实零分及确认失败零分；跳过 null 并重新归一化权重：

$$
\mathcal A^d_{i,k}=\{m:\ s_{i,k,m}\ne\mathrm{null},\ m\text{ 已启用且属于维度 }d\}.
$$

$$
S^d_{i,k}=\begin{cases}
\dfrac{\sum_{m\in\mathcal A^d_{i,k}}w_m s_{i,k,m}}{\sum_{m\in\mathcal A^d_{i,k}}w_m},&\mathcal A^d_{i,k}\ne\varnothing,\\
\mathrm{null},&\mathcal A^d_{i,k}=\varnothing.
\end{cases}
$$

## Case-level dimensions

$$
\mathcal K_i^d=\{k:\ S^d_{i,k}\ne\mathrm{null}\}.
$$

令 F_i^d 表示：在没有有效帧分数时，已有证据确认该维度因 Gen 未恢复而无法评分；包括整段无匹配目标的 Appearance。参考信息不足或原因不明不满足此条件。

$$
S_i^d=\begin{cases}
\dfrac{1}{|\mathcal K_i^d|}\sum_{k\in\mathcal K_i^d}S^d_{i,k},&|\mathcal K_i^d|>0,\\
0,&|\mathcal K_i^d|=0\ \land\ F_i^d=1,\\
\mathrm{null},&|\mathcal K_i^d|=0\ \land\ F_i^d=0.
\end{cases}
$$

默认启用三个核心维度。禁止删除未评分的启用维度来计算几何平均：

$$
M_i=\begin{cases}
0,&\exists d\in\{\mathrm{Inv},\mathrm{App},\mathrm{Spa}\}:S_i^d=0,\\
(S_i^{\mathrm{Inv}}S_i^{\mathrm{App}}S_i^{\mathrm{Spa}})^{1/3},&\text{三个维度均有分数},\\
\mathrm{null},&\text{仍有未确认的启用维度}.
\end{cases}
$$

已知一个核心维度为 0 时，总分确定为 0，但不会伪造其他未知维度的分数。
按配置关闭的维度不参与计算；仅启用部分核心维度时，几何平均指数为启用维度数的倒数。

## Benchmark

定义用于全样本统计的 case 分数：

$$
Q_i=\begin{cases}0,&g_i=0,\\M_i,&g_i=1.\end{cases}
$$

当全部 N 个 case 的 Q_i 都已确定时，报告：

$$
C_{\mathrm{gate}}=\frac1N\sum_{i=1}^N g_i,\qquad
\mathrm{Memory\ Score}=\frac{100}{N}\sum_{i=1}^N Q_i.
$$

通过 Gate 的均分作为辅助指标：

$$
\mathrm{Memory\ Score}_{\mathrm{passed}}=
\frac{100\sum_{i:g_i=1}M_i}{\sum_i g_i}.
$$

未解决的参考不足、未知原因或调用错误不能删出分母后伪装为完整 benchmark。存在未解决 case 时，主分数暂不发布；单独报告临时均分、已确定 case 数和待确认 case 数。所有 Gate 都失败时，主分数为 0，通过 Gate 的均分不适用。
