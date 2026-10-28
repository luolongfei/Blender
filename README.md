# Blender 角色制作工作区

本工作区管理我们的制作代码、工具版本和角色状态。Charakuru 是锁定版本的外部工具；模型和贴图通过文件复制迁移，不上传到普通 Git。

当前角色：`6c8500bf-1c38-481a-babd-e5478a50dd67`（双马尾绿裙角色）。

本次迁移清单生成于 **2026-10-28 17:56（北京时间）**：962 个资产文件，共约 7.54 GB，已在本机新目录完成全量 SHA-256 校验。当前主源和导出对应关系、已合入修改及剩余问题见 [角色状态](projects/6c8500bf-1c38-481a-babd-e5478a50dd67/project-state.json)。这是已保存成果的迁移快照，不表示原制作会话已暂停，也不是异机恢复或最终质量验收完成。

## 文件在哪里

| 位置 | 用途 | 保存方式 |
| --- | --- | --- |
| `scripts/`、`pipeline/`、`config/` | 启动、路径适配和配置 | Git |
| `toolchain.lock.json` | 完整工具包的版本、来源和校验值 | Git |
| `asset-manifests/` | 手动复制资产的路径、大小和 SHA-256 清单 | Git |
| `projects/<id>/work/local-model/` | 现有角色制作代码 | Git 中明确登记的源码 |
| `projects/<id>/` 中的模型、贴图、原画和过程产物 | 实际角色资产 | 手动复制并校验 |
| `tools/charakuru/<version>/` | 完整 Charakuru 运行套件 | 本机安装或手动复制 |
| `state/` | 本机应用状态与运行日志 | 本机保存 |

Git 中的代码和清单不能替代模型文件。角色主源固定为 `projects/<id>/work/character.blend`；新副本只有完成整合后才成为主源，不能根据修改时间自动替换。

## 开始使用

需要 Node.js 22 以上。工具包安装和 Python 源码语法检查还需要系统 Python 3.8 以上，只使用标准库；Blender 内置 Python 另计。当前已验证的制作环境基线是 Blender 5.2.3 LTS、VRM Add-on 4.10.2、Blender MCP 2.2.4；这些是恢复基线，不是“最新版”声明。

首次设置时，将 `config/local.example.json` 复制为 `config/local.json`，按需填写本机 Blender、Python 可执行文件和 Charakuru 配置路径；本机配置不入 Git。

在工作区根目录执行：

```sh
node scripts/workspace.mjs doctor
node scripts/assets.mjs verify --project 6c8500bf-1c38-481a-babd-e5478a50dd67
node scripts/workspace.mjs validate-source
```

缺少 Charakuru 工具包时：

```sh
node scripts/workspace.mjs install-tool
```

启动应用或独立审阅页面：

```sh
node scripts/workspace.mjs start
node scripts/workspace.mjs preview
```

首次启动会从已复制的默认角色 `project.charakit` 重建本机应用状态，不恢复旧电脑的排队任务。独立审阅入口使用当前 `exports/expression-review/` 中的模型。

工具命令成功不等于模型通过客户验收。以角色状态说明、对应导出报告和实际模型为准。

## 迁移与续作

阅读 [迁移步骤](docs/MIGRATION.md)，在新电脑拉取本仓库，再复制同一迁移阶段的资产和工具。本方案不需要 Git LFS。不要把旧 Charakuru 目录中的源码直接覆盖到这个工作区。

阅读 [制作管理规则](docs/PRODUCTION.md)，确认主源、待合并副本、导出是否与当前主源对应。当前 `.charakit` 的应用流程状态可能落后于 Blender 制作进度，不能据此重新生成或覆盖角色。

原制作会话的收尾请求在 [ORIGINAL-CHAT-REQUEST.txt](docs/ORIGINAL-CHAT-REQUEST.txt)。迁移准备时没有可用的跨会话发送工具，Codex 窗口控制也被工具限制拒绝，因此**这份文字需要在原会话中发送，不能把文件存在视为已经通知或停工**。

## 更新 Charakuru

使用专用完整 Release ZIP，不用 GitHub 的 Source code ZIP 代替运行套件。新版安装到新的版本目录，在角色副本上验证后再更新版本锁；当前工具和原始角色保留到验证完成。不要原地覆盖正在使用的工具版本。
