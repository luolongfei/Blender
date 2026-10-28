# 迁移到另一台电脑

本方案采用“Git 同步代码和清单，手动复制大文件”，不需要 Git LFS。**只执行 `git clone` 不会得到模型、贴图或工具安装文件。**

按以下顺序操作：旧电脑准备两个目录 → 新电脑安装 Git、Node.js 和 Python → clone `main` → 复制工具、导入并校验角色资产 → 安装 Blender、VRM Add-on 和 Blender MCP → 验收后续作。

软件的下载、安装、Codex MCP 配置和故障排查见 [新电脑环境安装](NEW_MACHINE_SETUP.md)。本页负责明确“搬什么、放哪里、怎样确认没有漏文件”。

## 1. 本次迁移基线

原制作会话已于北京时间 2026-10-28 18:15 完成交付。本工作区已接收最终交接，主源与当前导出没有版本差异；新增依赖审计报告和可选插件安装文件也已补齐。阅读 [已接收的交接](HANDOFF.md)，不必重复发送收尾请求。

本次角色资产清单为 963 个文件，合计 7,543,237,602 字节，约 7.54 GB。原会话已做本机换路径恢复测试，新电脑仍需实际验收。当前没有待合并的必需修复，模型仍是有已知视觉和性能问题的审阅版。

本工作区直接使用 `main`。搬运时保持新旧电脑使用相同的 Git 提交和资产清单。若旧工程之后又产生修改，应先核对并更新迁移快照，再搬运；不要同时在两台电脑修改同一个角色主源。

在旧电脑的 `/Users/luolongfei/dev/code/Blender` 中，确认本次代码、状态和清单已经提交并推送，再记录提交号：

```sh
git status --short --branch
git rev-parse HEAD
```

把完整提交号记录在传输目录旁的 `Blender-transfer/GIT-COMMIT.txt` 中，供新电脑比对。发现未提交修改或本地 `main` 领先远端时，先核对并完成本次快照的提交、推送，不把一套新资产配给另一版旧清单。

## 2. 旧电脑：哪些需要手动搬

以下“原电脑路径”都是当前已准备好的 **Blender 新工作区**，不要从旧 Charakuru 根目录混着取文件。“新电脑位置”相对于新电脑 clone 得到的 `Blender` 根目录。

| 内容 | 原电脑完整路径 | 新电脑位置 | 是否手动搬运 |
| --- | --- | --- | --- |
| 角色完整制作目录 | `/Users/luolongfei/dev/code/Blender/projects/6c8500bf-1c38-481a-babd-e5478a50dd67/` | `projects/6c8500bf-1c38-481a-babd-e5478a50dd67/` | **必须**，约 7.54 GB；先放暂存目录，再用第 5 节命令导入 |
| 固定版工具与插件安装文件 | `/Users/luolongfei/dev/code/Blender/tools/` | `tools/` | **强烈建议一起搬**，约 18.7 MB；可省去重新取得工具和插件载荷 |

必须搬整个角色目录。它不仅有当前 `.blend`、`.vrm`，还保留原画、贴图、动作、历史检查点、工作副本和报告。不要只挑 `exports/`、一个 VRM 或一个 `.charakit`。

`tools/` 内应保持以下结构：

```text
tools/
  charakuru/
    1.1.0/                         固定版 Charakuru 完整运行套件
  setup/
    VRM_Addon_for_Blender-Extension-4_10_2.zip
    mcp_for_blender-2.2.4-py3-none-any.whl
    blender-mcp-2.2.4/
      addon.py
```

这些是工具文件和安装载荷，**复制不等于在新机完成安装**。MCP 的 Python 依赖没有全部离线打包；安装时仍可能需要网络。

### 外置盘或传输目录怎么摆

例如建立名为 `Blender-transfer` 的目录，把原电脑的角色目录和 `tools` 按下面的层级复制进去：

```text
Blender-transfer/
  projects/
    6c8500bf-1c38-481a-babd-e5478a50dd67/
      project.charakit
      work/
      exports/
      textures/
      images/
      models/
      previews/
      ...                          原角色目录中的其余文件也保留
  tools/
    charakuru/
      1.1.0/
    setup/
      ...
```

角色目录里也有少量 Git 管理的源码，整目录复制到暂存区没有问题。新电脑使用第 5 节的资产导入命令，会保留 clone 得到的适配后源码。

### 哪些已经在 Git，不必手动搬

`scripts/`、`pipeline/`、`config/` 中的公共配置和样例、`docs/`、`asset-manifests/`、`toolchain.lock.json`、根目录说明，以及角色登记的制作源码和 `project-state.json`，都随 `main` 获取。

### 哪些不要照搬

| 内容 | 处理方式 |
| --- | --- |
| 旧 Charakuru 或旧工作区的 `.git/` | 不覆盖新仓库；新电脑用 clone 创建 Git 元数据 |
| `state/` | 通常不用搬；应用状态首次启动时重建。它不是角色资产正本 |
| `config/local.json`、`config/charakuru.local.json` | 新机按样例重新填写，避免旧电脑路径失效 |
| `.codex/`、Codex 登录数据、浏览器 Cookie/配置 | 不整份复制；在新机正常安装、登录及配置需要的 MCP |
| `node_modules/`、虚拟环境、Python 缓存、`.DS_Store` | 不作为迁移资料；按需重建 |
| 旧电脑的 Blender/Node 应用程序本体 | 按新电脑操作系统和 CPU 架构安装 |

如需保留历史日志和交接排错资料，可选把 `/Users/luolongfei/dev/code/Blender/state/` 单独备份到传输盘的 `Blender-transfer/optional-state-backup/`，不提交到 Git。新机默认不恢复为活动 `state/`，仍从 `.charakit` 重建；尤其不要恢复旧排队请求。它是历史后备，不是继续制作的必需目录。

### 旧完整换机包是否也要搬

下面两份文件可作为额外恢复后备，但不是上述“两目录”方案的必需文件，不用为了续作重复搬两套相同资产：

- `/Users/luolongfei/dev/code/Charakuru-delivery/Charakuru-workspace-2026-10-28.zip`，约 6.24 GB。
- `/Users/luolongfei/dev/code/Charakuru-delivery/Charakuru-workspace-2026-10-28.zip.sha256`。

如果选择保留这个备份，应把 ZIP 和校验文件一起搬。旧 ZIP 包含旧 `.git`、根目录代码、配置和会话状态，**不能整体解压覆盖新的 Blender 工作区**。其中的操作说明针对旧目录布局；日常续作以本仓库文档为准。

`character-review-2026-10-28.zip` 只是审阅包，不足以替代完整制作目录。

## 3. 新电脑：先准备基础软件并取得代码

先按 [新电脑环境安装](NEW_MACHINE_SETUP.md) 安装 Git、Node.js 和 Python。后续 MCP 自身要求 Python 3.10 以上；本工作区的工具安装和语法检查最低只要求 3.8。Node.js 要求 22 以上。

在新电脑希望存放项目的父目录打开终端，执行：

```sh
git clone --branch main https://github.com/luolongfei/Blender.git
cd Blender
git rev-parse HEAD
```

最后一条命令显示 Git 提交号，应与 `Blender-transfer/GIT-COMMIT.txt` 中记录的迁移提交一致。若不一致，先核对远端提交和资产快照，不能重写清单消除差异。macOS 的新根目录可以是 `/Users/<新用户名>/dev/Blender`，Windows 可以是 `D:\dev\Blender`；不要求沿用原电脑用户名或绝对路径。

此后本页所有 `node scripts/...` 命令都在**新电脑的 Blender 工作区根目录**执行。

现在就按 [本机路径配置](NEW_MACHINE_SETUP.md#3-取得代码资产与本机路径配置) 从 `config/local.example.json` 创建 `config/local.json`，先填写实际 `pythonExecutable`，再执行后面的安装或语法检查命令。使用 uv 时以 `uv python find 3.12` 的输出为准。此时 Blender 尚未安装可暂留空，装好后补填；不要等到第 6 节才配置 Python。

## 4. 新电脑：放好工具

从暂存目录把整个 `Blender-transfer/tools/` 复制到新工作区的 `tools/`，确保得到 `Blender/tools/charakuru/1.1.0/`，而不是多套一层 `Blender/tools/tools/`。

不要把旧完整 ZIP 的 Charakuru 根目录当作这个固定版工具目录复制，也不要用 GitHub 的 Source code ZIP 替代完整工具套件。

如果没有手动搬运固定版 Charakuru，可在新工作区执行：

```sh
node scripts/workspace.mjs install-tool
```

这个命令只安装锁定版 Charakuru；不会代替 Blender、VRM Add-on、MCP 或 `tools/setup/` 安装载荷的恢复。后者按 [新电脑环境安装](NEW_MACHINE_SETUP.md) 操作。

## 5. 新电脑：导入并校验角色资产

把下面 `--source` 后面的路径替换成实际外置盘或暂存目录中的**角色目录**。该目录应直接包含 `project.charakit`、`work`、`exports` 等，不要填它的上一级 `projects`。

macOS 示例，外置盘名假设为 `Transfer`：

```sh
node scripts/assets.mjs copy --source "/Volumes/Transfer/Blender-transfer/projects/6c8500bf-1c38-481a-babd-e5478a50dd67" --project 6c8500bf-1c38-481a-babd-e5478a50dd67
```

Windows PowerShell 示例，外置盘盘符假设为 `E:`：

```powershell
node scripts/assets.mjs copy --source "E:\Blender-transfer\projects\6c8500bf-1c38-481a-babd-e5478a50dd67" --project 6c8500bf-1c38-481a-babd-e5478a50dd67
```

两种系统导入后都执行：

```sh
node scripts/assets.mjs verify --project 6c8500bf-1c38-481a-babd-e5478a50dd67
node scripts/workspace.mjs doctor
node scripts/workspace.mjs validate-source
```

本次基线的资产验证应显示 `status: "ok"`、`checkedFiles: 963`、`extraFiles: 0`、`failures: []`。`doctor` 应确认固定工具和主 Blend 存在；它不验证 Blender MCP 是否已连接。`validate-source` 只做源码语法检查，不会运行建模补丁或证明模型外观合格。

复制命令保护 Git 登记的源码。目标已有不同内容时会拒绝覆盖并报告，先核对文件来源；不要直接加 `--refresh` 或重新生成清单把问题掩盖掉。校验失败时保留两端文件，处理缺失或不一致项后重新校验。

**新电脑第一次恢复时只执行 `verify`，不要先执行 `inventory`。** `inventory` 会把当前文件重新写成基准清单，不能用来证明搬运完整。

## 6. 新电脑：完成软件安装和恢复验收

按 [新电脑环境安装](NEW_MACHINE_SETUP.md) 完成 Blender、VRM Add-on、Blender MCP 两端和 Codex 配置，补齐第 3 节创建的 `config/local.json` 中的 Blender 路径。所有可执行文件路径使用新电脑实际位置，已有配置不覆盖。

随后依次验收：

1. 用 Blender 打开 `projects/6c8500bf-1c38-481a-babd-e5478a50dd67/work/character.blend`，这是唯一建模主源。
2. 检查贴图、骨架、表情和物理设置；确认 MCP 能读取正确场景。
3. 运行 `node scripts/workspace.mjs preview`，按打印的地址检查当前模型的静态状态和短动作。
4. 在 Codex 中打开新的 `Blender` 工作区，把 [续作提示词](CONTINUE_PROMPT.txt) 发给新会话。
5. 恢复验收通过后才继续精修。旧电脑原件和传输副本保留到新机恢复成功。

`finish_comprehensive_refinement.py` 等本轮变形补丁已经执行，不能对当前主源重复运行。`.charakit` 显示较早的制作阶段，不表示 Blender 模型丢失；不要为恢复进度重新触发 Tripo 候选生成。

## 后续又有资产修改时

本次资产清单已经生成并校验，直接搬运本次快照不必再生成。以后需要制作新的迁移快照时，由制作端先保存、停止写入，再执行：

```sh
node scripts/assets.mjs inventory --project 6c8500bf-1c38-481a-babd-e5478a50dd67
node scripts/assets.mjs verify --project 6c8500bf-1c38-481a-babd-e5478a50dd67
```

将更新后的角色状态和清单提交到 `main`，再搬运对应资产。清单生成或 Git 推送成功不等于大文件已经复制到另一台电脑。

新版 Charakuru 并列安装，在角色副本上验证打开、保存、材质、表情、物理与导出后再切换工具锁，不覆盖当前唯一可用工具版本。
