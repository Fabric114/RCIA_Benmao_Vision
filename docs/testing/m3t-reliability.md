# M3T 跟踪可靠性：操作说明与验收清单

状态：文档草案。自动测试与实机验收分开记录。本文件中的实机项均为“未执行”，不能据此宣称实机问题已修复。

依据：`docs/superpowers/specs/2026-11-20-m3t-reliability-and-replay-design.md`（第 10 节为 2026-11-21 批准的方案 B 变更）与 `docs/superpowers/plans/2026-11-20-m3t-state-machine.md`。

## 1. 自动测试

在仓库根目录执行：

```bash
cd /home/shaozi/RCIA_Benmao_Vision
python3 -m unittest tests.test_latest_rgbd_buffer tests.test_tracking_watchdog_validation tests.test_m3t_tracking_health_gate tests.test_tracking_state_filter tests.test_simple_tracking_lifecycle tests.test_supervisor_message_routing tests.test_tracking_health_debug_analyzer -v
source /opt/ros/humble/setup.bash
python3 -m unittest tests.test_tracking_supervisor_node_helpers tests.test_tracking_supervisor_lifecycle_integration tests.test_tracking_reliability_launch -v
```

- 第一条覆盖纯 Python 几何/状态机/生命周期/消息路由/诊断分析器模块。
- 第二条覆盖 ROS 节点 helper、集成流程与 launch 参数绑定（需要 ROS 2 Humble 环境，缺环境时不能标记为全部通过）。
- 分析器没有命令行入口，需在 Python 中调用 `analyze_tracking_health(payload, previous_payload)`，返回 `should_emit`、`primary_cause`、`summary`、`evidence`。

## 2. 部署与重启

- 修改 `scripts/` 下的 Python 后，必须重启 supervisor，正在运行的进程不会热更新。
- `ros2/` 下的节点或 launch 若通过复制安装，需要重新构建或安装该包，并更新安装产物。
- 部署时只更新本次相关源码，不同步覆盖远端 M3T 缓存。

## 3. 运行中的检查

1. 启动后查看日志中的有效参数，确认期限参数与下表一致。
2. 订阅 `/energy/tracking_state` 与 `/energy/tracking_health`，检查 `reliability_state`、`reason`、`stop_m3t`、`request_reinit`。
3. 状态进入 SUSPECT、LOST、NO_TRACK 或 REINIT_NEEDED 后，下游不能继续把最后一条位姿当作实时数据。
4. 出现 `stop_ack_timeout` 时，系统应等待 stop 确认，不得重叠启动新的初始化。

## 4. 参数迁移表

下表对应 `ros2/rcia_energy_trace_ros/launch/full_energy_tracking.launch.py` 与 `ros2/rcia_energy_trace_ros/scripts/tracking_supervisor_node.py` 的当前已接入默认值。

| launch 参数 | 节点参数名 | 当前默认值 | 状态 | 说明 |
|---|---|---:|---|---|
| `supervisor_max_yolo_age_ms` | `max_yolo_age_ms` | 500.0 | 保留 | YOLO 存在性的新鲜度上限（ms） |
| `stable_frames` | `stable_yolo_frames` | 3 | 保留 | 消失确认后再次初始化所需的连续稳定 YOLO 帧数 |
| `supervisor_tracking_min_depth_ratio` | `tracking_min_depth_consistency_ratio` | 0.60 | 保留 | 深度支持门槛 |
| `supervisor_min_yolo_projection_iou` | `min_yolo_projection_iou` | 0.35 | 保留 | 本期不调整 |
| `filter_owned_confirm_frames` | `filter_owned_confirm_frames` | 3 | 保留 | 连续新观测确认接管 |
| `filter_recover_frames` | `filter_recover_frames` | 2 | 保留 | SUSPECT 中连续新正面证据数 |
| `filter_min_rendered_pixels_for_owned` | `filter_min_rendered_pixels_for_owned` | 40 | 保留 | 仅作为投影出画界限，不作为正面支持 |
| `filter_out_of_view_frames` | `filter_out_of_view_frames` | 12 | 弃用 | 已由 `filter_absence_timeout_ms` 取代，不再控制期限 |
| `filter_reinit_conflict_frames` | `filter_reinit_conflict_frames` | 5 | 弃用 | 已由 `filter_conflict_timeout_ms` 取代，只用于 M3T_STARTING |
| `filter_suspect_frames` | `filter_suspect_frames` | 6 | 弃用 | 已由 `filter_suspect_timeout_ms` 取代 |
| `filter_bad_tracking_frames` | `filter_bad_tracking_frames` | 5 | 弃用 | 不再控制期限 |
| `filter_startup_grace_frames` | `filter_startup_grace_frames` | 40 | 弃用 | 已由 `filter_startup_timeout_ms` 取代 |
| `filter_occlusion_grace_frames` | `filter_occlusion_grace_frames` | 24 | 弃用 | 不再控制期限 |
| `start_frame_tolerance_ms` | （M3T C++ 节点） | 80.0 | 待 B 阶段 | 初始帧容差需与 bridge 和 C++ 默认值统一，本期不改 |

以下为本次新增并已接入 launch 与 `tracking_supervisor_node.py` 的参数：

| 参数 | 默认值 | 单位 | 说明 |
|---|---:|---|---|
| `filter_absence_timeout_ms` | 750.0 | ms | 新鲜 YOLO 连续缺席持续的期限，到期强制 LOST（`out_of_view`），即使背景深度与旧位姿一致 |
| `filter_conflict_timeout_ms` | 625.0 | ms | 仅 M3T_STARTING：目标可见且投影冲突，到期 REINIT_NEEDED |
| `filter_suspect_timeout_ms` | 1000.0 | ms | SUSPECT 最长时间，到期 LOST（`observation_unavailable` 或 `support_timeout`） |
| `filter_startup_timeout_ms` | 5000.0 | ms | 发布初值至取得可信跟踪的总期限 |
| `tracking_yolo_frame_tolerance_ms` | 80.0 | ms | YOLO 与位姿帧投影匹配的最大时间差 |
| `tracking_pose_frame_tolerance_ms` | 15.0 | ms | 健康检查与位姿帧的最大时间差 |
| `disappearance_confirm_frames` | 3 | 帧 | NO_TRACK 中连续无目标帧数，确认消失后才允许再次初始化 |
| `filter_hard_depth_mismatch_ratio` | 0.05 | 比例 | 有足够有效深度时，低于等于此值为强深度反证 |
| `m3t_stop_ack_timeout_ms` | 2000.0 | ms | stop 确认等待上限，超时报告 `stop_ack_timeout` 并保持等待 |

已删除：`no_target_support_timeout_ms`（原 500 ms），不再使用。

## 5. 实机验收清单

全部项目目前均为未执行。每个场景至少进行 10 次，逐次记录结果。

| 编号 | 场景 | 次数 | 记录项 | 结果 |
|---|---|---:|---|---|
| R1 | 静止出画：目标离开画面，背景假框保持 | ≥10 | 第一观测时间、最后可信输出、LOST 时间、stop、stopped、恢复时间 | 未执行 |
| R2 | 背景假框：目标不在画面，背景深度与旧位姿一致 | ≥10 | 同 R1，并确认 `reason=out_of_view` | 未执行 |
| R3 | 短暂遮挡后恢复：遮挡约 250 ms | ≥10 | SUSPECT 次数、是否 stop、是否 reinit、恢复到 OWNED 时间 | 未执行 |
| R4 | YOLO 漏检但深度良好 | ≥10 | 是否保持 OWNED、是否误触发 REINIT_NEEDED | 未执行 |
| R5 | 深度断流或有效重叠不足 | ≥10 | `reason=observation_unavailable` 或 `support_timeout`，1000 ms 内 LOST | 未执行 |
| R6 | 目标重入画面 | ≥10 | 3 帧消失确认、3 帧稳定 YOLO 后的初始化请求，是否重叠启动 | 未执行 |
| R7 | 启动无进展 | ≥3 | `startup_timeout` 在 5000 ms 触发，仅一次 stop | 未执行 |
| R8 | stop 确认超时 | ≥3 | `stop_ack_timeout` 出现，确认前无新初始化 | 未执行 |
| R9 | 稳定期间的 SUSPECT 发布 | ≥3 | SUSPECT 期间 `/energy/trusted_pose` 无新增发布 | 未执行 |

记录模板：

| 日期 | 场景编号 | 次数序号 | 第一观测时间 | 最后可信输出 | stop | stopped | 重初始化请求 | 恢复时间 | 备注 |
|---|---|---:|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |

## 6. 未承诺项

- 不承诺 0.15～0.2 s 追帧完成，本文件也不包含该项的实测数据。
- 不承诺绝对零漂移或实物精度。
- 外部触发话题（MCU 串口信号）为未实现的扩展点。本期未新增话题、消息或参数。
