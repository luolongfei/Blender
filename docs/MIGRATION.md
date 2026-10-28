# 迁移到另一台电脑

本方案采用“Git 同步代码和清单，手动复制大文件”，不需要 Git LFS。迁移源应是本工作区准备好的同一阶段快照；不要把仍在修改的旧目录当作已冻结备份。

## 1. 确认本次交接基线

原制作会话已于北京时间 2026-10-28 18:15 完成交付。本工作区已接收其最终交接，主源与当前导出没有版本差异；新增依赖审计报告和可选插件安装文件也已补齐。阅读 [已接收的交接](HANDOFF.md)，不必重复发送收尾请求。

本次迁移可以直接从新 Blender 工作区取文件。当前没有待合并的必需修复，模型仍是有已知视觉和性能问题的审阅版。原会话已做本机换路径恢复测试，新电脑尚待验证。

旧完整 ZIP 保留作恢复后备；其中的旧 `.git`、根代码和会话状态不能覆盖本工作区。若旧工程之后继续产生新修改，应另行核对并更新迁移快照。

## 2. 准备需要手动搬运的目录

| 目录 | 是否需要搬运 | 说明 |
| --- | --- | --- |
| `projects/6c8500bf-1c38-481a-babd-e5478a50dd67/` | 必须 | 包含主源、图片、纹理、过程副本和输出 |
| `tools/` | 推荐 | 包含固定版 Charakuru，以及 `tools/setup/` 中的可选 VRM/MCP 安装文件 |
| `state/` | 通常不用 | 本机状态可重建，不作为角色成果依据 |

角色目录可能同时包含 Git 管理的源码和手动管理的资产。可以把整个角色目录放到外置盘或临时传输目录，但在新机使用下面的资产复制命令恢复，避免覆盖 Git 中已适配的源码。

工具目录必须来自本工作区，保持 `tools/charakuru/<version>/` 和 `tools/setup/` 层级。不要用上游仓库的 Source code ZIP 或旧完整备份的根目录替代锁定工具套件。

本次资产清单已经生成并校验。直接搬运该快照时不必重新生成；后续又有资产修改时，在停止写入后执行：

```sh
node scripts/assets.mjs inventory --project 6c8500bf-1c38-481a-babd-e5478a50dd67
node scripts/assets.mjs verify --project 6c8500bf-1c38-481a-babd-e5478a50dd67
```

清单随代码提交到 Git。实际资产仍需要复制到外置盘、局域网目标或新电脑；清单生成成功不等于备份完成。

## 3. 新电脑先取得代码

本工作区直接使用 `main` 分支，代码、版本锁和迁移清单都保存在该分支。

```sh
git clone --branch main https://github.com/luolongfei/Blender.git
cd Blender
git rev-parse HEAD
```

将显示的提交号与源端迁移提交核对。两台电脑应使用同一份代码、工具版本锁和资产清单。

安装 Node.js 22 以上和系统 Python 3.8 以上。Python 用于工具包解压与 Python 源码语法检查，只使用标准库，不需要额外安装 Python 包。它与 Blender 内置 Python 分开管理。Blender、VRM Add-on 和 Blender MCP 需要在新电脑上安装并连接，不能由项目文件复制代替。恢复时先使用版本锁记录的已验证版本。

如果手动复制工具，把整个 `tools/` 放到新工作区同名目录。没有复制固定版 Charakuru 时也可执行：

```sh
node scripts/workspace.mjs install-tool
```

`tools/setup/` 的安装文件来源和校验值记录在版本锁中，需要在新机实际安装。MCP 2.2.4 需要 Python 3.10 以上，相关 Python 依赖没有全部离线打包。Blender 和 Node 本体仍需按新机系统安装。

首次设置时，将 `config/local.example.json` 复制为 `config/local.json`。macOS / Linux 可执行：

```sh
cp -n config/local.example.json config/local.json
```

已有配置时直接编辑，不覆盖它。Windows 可以通过文件管理器复制。这个文件被 Git 忽略，按需填写：

- `blenderExecutable`：本机 Blender 可执行文件的完整路径。
- `pythonExecutable`：本机 Python 3.8 以上可执行文件的完整路径；留空时尝试环境配置及 `python3`、`python`。
- `charakuruConfig`：可选的 Charakuru 本机配置文件路径；相对路径按工作区根目录解析。

macOS 与 Windows 的可执行文件位置不同。公共配置保持相对路径，不要把旧机器路径写回通用制作代码。

## 4. 从传输目录恢复大文件

在新工作区根目录执行，将占位文字替换成外置盘或暂存目录中的角色目录：

```sh
node scripts/assets.mjs copy --source "<传输目录中的角色目录>" --project 6c8500bf-1c38-481a-babd-e5478a50dd67
node scripts/assets.mjs verify --project 6c8500bf-1c38-481a-babd-e5478a50dd67
```

复制命令用于资产恢复，并保护登记的源码。目标已有不同内容时，先核对哪份正确，不要随意覆盖。校验失败时保留两端文件，处理缺失或不一致项后重新校验。

工具和资产恢复后：

```sh
node scripts/workspace.mjs doctor
node scripts/workspace.mjs validate-source
```

## 5. 验证能继续工作

1. 核对交接中的主源，确认不会重跑已经执行过的变形补丁。
2. 用 Blender 打开主源，检查贴图、骨架、表情和物理设置。
3. 确认 MCP 能读取正确场景。
4. 检查短动作或阶段说明要求的关键姿势，确认导出与主源的对应关系。
5. 验证后再继续未完成修正；旧电脑原件保留到新电脑恢复成功。

新电脑在 Codex 中打开的是这个 `Blender` 工作区，把 [续作提示词](CONTINUE_PROMPT.txt) 发给新会话。它已适配本仓库的路径、命令、`main` 分支和最终交接状态。

`.charakit` 显示较早的阶段，不表示 Blender 模型丢失。不要为“恢复进度”重新触发候选生成，应先恢复制作源，再核对应用登记。

## 更新工具时

下载新版到另一个版本目录，在可丢弃的角色副本上检查打开、保存、材质、表情、物理与导出。通过后再更新工具锁和角色状态，不直接覆盖当前唯一可用的工具包。
