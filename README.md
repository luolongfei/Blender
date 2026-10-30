# Blender 角色制作工作区

本工作区管理我们的制作代码、工具版本和角色状态。Charakuru 是锁定版本的外部工具；模型和贴图通过文件复制迁移，不上传到普通 Git。

当前角色：`6c8500bf-1c38-481a-babd-e5478a50dd67`（双马尾绿裙角色）。

当前制作进度：按用户新提供的 `character-hq-textured.glb` 完成静态动漫重建交付，主源已更新为本轮可编辑工程，原主源先复制并校验保存。交付位于 `Character_Rebuild/`；GLB已通过空场景重导入、贴图与逐网格内部接触检查，无骨骼或动画。面部神态、发束和细衣褶仍有简化，与原画精细程度有差距，未获客户外观验收。旧R11制作保持停止，旧VRM和缓存均为历史。详见 [最新交接](docs/HANDOFF.md)。

新增网页动作预览：`Character_Rebuild/interactive/viewer.html` 使用本次角色的独立绑定副本，支持旋转、缩放、平移和站立、招手、转头、原地轻踏。22骨骼、4段动作及T姿势已导出并重新导入验证；原静态交付和主源保持不变。启动：`node Character_Rebuild/scripts/serve_delivery.mjs --port 52860`，打开 `http://127.0.0.1:52860/interactive/viewer.html`。当前没有表情、独立指骨、头发或裙子物理模拟。

Windows 续接状态（2026-10-28）：资产已恢复；Windows Blender 5.2.3 LTS、VRM 4.10.2 和 MCP 2.2.4 已安装，通过正确主源的 MCP 读取检查。详见 [Windows 安装验收记录](docs/WINDOWS_SETUP_RESULT.md)。模型外观和客户验收状态不变。

历史迁移基线（不覆盖本轮重建）：已接收原制作会话的最终交接。该迁移清单更新于 **2026-10-28 18:38（北京时间）**：963 个资产文件，共约 7.54 GB，已完成全量 SHA-256 校验，全部资产条目与原交接包清单一致。这是当时的恢复结果；此后主源已更新，新增独立副本与本轮交付不在旧清单中。详见 [交接说明](docs/HANDOFF.md) 和 [角色状态](projects/6c8500bf-1c38-481a-babd-e5478a50dd67/project-state.json)。最终质量验收尚未完成。

## 文件在哪里

| 位置 | 用途 | 保存方式 |
| --- | --- | --- |
| `scripts/`、`pipeline/`、`config/` | 启动、路径适配和配置 | Git |
| `toolchain.lock.json` | 完整工具包的版本、来源和校验值 | Git |
| `asset-manifests/` | 手动复制资产的路径、大小和 SHA-256 清单 | Git |
| `Character_Rebuild/` 的脚本、网页、说明和小型检查报告 | 本轮重建与动作预览源码、证据 | Git；模型、图片和 ZIP 除外 |
| `projects/<id>/work/local-model/` | 现有角色制作代码 | Git 中明确登记的源码 |
| `projects/<id>/` 中的模型、贴图、原画和过程产物 | 实际角色资产 | 手动复制并校验 |
| `tools/charakuru/<version>/`、`tools/setup/` | 固定版工具和可选插件安装文件 | 本机安装或手动复制 |
| `state/` | 本机应用状态与运行日志 | 本机保存 |

Git 中的代码和清单不能替代模型文件。角色主源固定为 `projects/<id>/work/character.blend`；新副本只有完成整合后才成为主源，不能根据修改时间自动替换。

## 开始使用

新电脑从零安装请先看 [新电脑环境安装与验收](docs/NEW_MACHINE_SETUP.md)：包含 Git、Node.js、Python/uv、Blender、VRM Add-on、Blender MCP 两端及 Codex 配置，分别提供 macOS 和 Windows 示例。

需要 Node.js 22 以上。工具包安装和 Python 源码语法检查还需要系统 Python 3.8 以上，只使用标准库；Blender 内置 Python 另计。当前已验证的制作环境基线是 Blender 5.2.3 LTS、VRM Add-on 4.10.2、Blender MCP 2.2.4；这些是恢复基线，不是“最新版”声明。

首次设置时，将 `config/local.example.json` 复制为 `config/local.json`，按需填写本机 Blender、Python 可执行文件和 Charakuru 配置路径；本机配置不入 Git。

在工作区根目录执行：

```sh
node scripts/workspace.mjs doctor
node scripts/workspace.mjs validate-source
```

旧 `assets:verify` 仅适用于对应的历史迁移快照，不能用于认定本轮重建已备份。不要为让旧清单通过而覆盖当前主源或重写清单。

缺少 Charakuru 工具包时：

```sh
node scripts/workspace.mjs install-tool
```

启动应用或独立审阅页面：

```sh
node scripts/workspace.mjs start
node scripts/workspace.mjs preview
```

首次启动会从已复制的默认角色 `project.charakit` 重建本机应用状态，不恢复旧电脑的排队任务。上面的旧 `preview` 命令仍展示历史 `exports/expression-review/`，不对应本轮静态重建。

本轮交付图册为 `Character_Rebuild/review.html`。运行 `node Character_Rebuild/scripts/serve_delivery.mjs` 可开启仅本机访问的网页，输出地址也记录在 `Character_Rebuild/reports/preview-server.json`；该图册包含本轮模型下载、实际检查图和同条件前后对照。

工具命令成功不等于模型通过客户验收。以角色状态说明、对应导出报告和实际模型为准。

## 迁移与续作

阅读 [迁移步骤](docs/MIGRATION.md)，在新电脑拉取本仓库，再复制同一迁移阶段的资产和工具。**必须手动搬运整个角色目录、`Character_Rebuild/` 的本地资产及根目录原输入 `character-hq-textured.glb`，强烈建议同时搬运 `tools/`**；文档列有旧机完整路径、新机目标位置和导入命令。Git clone 不包含模型与贴图。本方案不需要 Git LFS。不要把旧 Charakuru 目录中的源码直接覆盖到这个工作区。

阅读 [制作管理规则](docs/PRODUCTION.md)，确认主源、待合并副本、导出是否与当前主源对应。当前 `.charakit` 的应用流程状态可能落后于 Blender 制作进度，不能据此重新生成或覆盖角色。

原制作会话已完成交付，不必再发送收尾请求。新电脑在 Codex 中打开本工作区，使用 [续作提示词](docs/CONTINUE_PROMPT.txt)。原 6.24 GB 完整 ZIP 保留作后备，其中的旧 Git、根目录代码和会话配置不能覆盖本工作区。

## 更新 Charakuru

使用专用完整 Release ZIP，不用 GitHub 的 Source code ZIP 代替运行套件。新版安装到新的版本目录，在角色副本上验证后再更新版本锁；当前工具和原始角色保留到验证完成。不要原地覆盖正在使用的工具版本。
