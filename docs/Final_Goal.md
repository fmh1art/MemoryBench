- **Goal**：评估 ACWM 是否能够利用历史信息，在当前不可直接观测的情况下，正确维持、更新并恢复世界状态。

- **Design Space**：每个 Memory case 由三个维度定义：
  - **Memory Operation**：Write、Maintain、Update，描述模型如何操作记忆。
  - **State Type**：Object Existence、Attribute / State、Spatial Location、Object Relation、Scene Modification，描述模型需要记住什么。
  - **Stress Dimension**：Delay、Memory Load、Interference、Similarity、Update Depth，描述任务如何逐渐变难。

- **Task Families**：将 Design Space 实例化为具体任务，包括 Hide & Reveal、Relocation、State Toggle、Persistent Scene Edit 和 Multi-object Assignment。

- **Case Definition**：一个 case 由 `Design Space Point + Task Family + Protocol` 构成，其中 Design Space Point 对应一个具体的 `Operation × State Type × Stress` 组合。

- **Evaluation Protocol**：每个 case 通常包含四个阶段：
  1. Initialize：建立初始场景与状态。
  2. Memory Event：发生需要模型记住的关键事件。
  3. Retention / Update：经过延迟、干扰或新的状态更新。
  4. Probe：重新暴露相关区域，并评估模型生成结果。

- **Ground Truth**：重点标注与 Memory 直接相关的状态，包括 Object Existence、Identity、Location、Relation 和 State / Attribute，而不是要求整个视频像素级一致。

- **Metrics**：
  - **Task Success**：Memory 任务是否整体完成。
  - **State Fidelity**：生成状态与 GT 的一致程度。
  - **Exact Match**：关键状态是否全部正确。
  - **Error Breakdown**：统计 Forgotten、Hallucinated、Stale、Misbound、Duplicated 等典型错误。

- **Reporting**：从多个角度形成模型的 Memory 能力画像，包括 Operation Score、State-type Score、Stress Curve、Error Breakdown 和典型 Case Study。

- **Example**：对于 “Apple in Drawer”，模型先观察苹果被放入抽屉，经过一段时间后再次打开抽屉；评估生成视频中苹果是否仍然存在、身份是否正确、位置是否与 GT 一致。

- **Summary**：Memory Benchmark 系统评估 ACWM 在不同状态类型和不同记忆压力下，对历史世界状态进行 Write、Maintain 和 Update，并在后续重新观测时正确恢复状态的能力。