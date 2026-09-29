# Task Families

| 任务族 | Memory Event | Probe 要检查什么 |
| --- | --- | --- |
| Hide & Reveal | 物体被遮挡或放入容器 | 再次揭示时的存在性、身份和位置 |
| Relocation | 物体从 A 移到 B | 恢复最新位置，避免使用旧位置 |
| State Toggle | 物体开合、开关等状态变化 | 恢复最新状态 |
| Persistent Scene Edit | 持久改变场景布局或内容 | 变化在后续视角中仍存在 |
| Multi-object Assignment | 多物体分别放到指定位置 | 身份与位置一一对应，没有错绑或重复 |

任务族与 Operation / State Type / Stress 独立标记，同一任务族可以有不同压力配置。
现有脚本只执行人工选好的视频样本，没有实现五类任务的自动生成器；扩展样本时从 `case_definition/` 接入。
