# 新电脑环境安装与验收

本页适用于从零安装的新电脑，与 [迁移步骤](MIGRATION.md) 配合使用。先准备基础软件，取得 Git 中的 `main` 和手动搬运的资产，再安装 Blender 插件、连接 MCP，最后验收。**本页是安装指南；新电脑尚未实际恢复或验收。**

以下版本来自本项目的 [工具锁](../toolchain.lock.json)，用于恢复已知环境，不代表应该追随最新版。联网安装仍可能获取新的 Python 间接依赖；我们没有保存完整依赖锁和所有依赖 wheel，因此不能承诺完全离线或逐字节相同的软件环境。

## 1. 要安装什么

| 软件或组件 | 本项目要求 | 用途与安装来源 |
| --- | --- | --- |
| Git | 安装适合新系统的版本 | 获取代码、清单和文档；[官方下载](https://git-scm.com/install/) |
| Node.js | **22 以上**；可选官方仍受支持的 LTS | 启动 Charakuru、预览、复制和校验资产；[官方下载](https://nodejs.org/en/download) |
| uv | 旧机验证版本 **0.13.5** | 管理独立的 Python 和 MCP 环境；按第 2 节安装 |
| 系统 Python | MCP 要求 **3.10 以上**；本指南示例安装 **3.12** | 工作区脚本与 MCP 使用；不是 Blender 内置 Python |
| Blender | 恢复基线 **5.2.3 LTS** | 打开和编辑主源；从 [官方版本入口](https://www.blender.org/download/previous-versions/) 选择该版本及新机系统、CPU 架构 |
| VRM Add-on for Blender | **4.10.2** | VRM、MToon、表情与物理数据；安装搬来的固定 ZIP |
| Blender 端 MCP 插件 | 随 **mcp-for-blender 2.2.4** 保存的 `addon.py` | 在 Blender 中接收本地控制连接 |
| 主机端 MCP 服务 | Python 包 **mcp-for-blender==2.2.4** | 由 Codex 启动，通过 STDIO 提供工具，并连接 Blender 插件 |
| Codex 所在的桌面应用 | 按新机平台安装并正常登录 | 打开本工作区、配置本地 MCP；[官方安装入口](https://developers.openai.com/codex/app)可能显示为 ChatGPT 桌面应用，在其中选择 Codex |
| 浏览器 | Chrome 或能打开本地预览的浏览器 | 查看 `127.0.0.1` 上的应用和角色预览 |

工作区的工具安装和源码语法检查实际只要求 Python 3.8 以上；选用 3.12 是为统一满足 MCP 的最低要求，**不是声称本项目已在新机 Python 3.12 上验证**。旧机 MCP 使用的 Python 3.15.1 记录在锁文件中；不必仅为一致的版本号强行安装它。

当前续作是已有模型的本地 Blender 路线，只需要上述 Blender MCP。Tripo、付费生成服务、图片生成工具、其他浏览器 MCP、VS Code 和 GitHub CLI 都不是恢复必需项。`ffmpeg` 仅在后续需要生成或转换视频时另装；打开工程、编辑模型和导出 VRM 不依赖它。需要代理操作桌面或浏览器时，在桌面应用里按实际功能配置相应系统权限，这与 Blender MCP 是两条独立连接。

**不用执行 `npm install`。** 本仓库的启动、迁移脚本及锁定的 Charakuru 常规启动不需要安装 npm 依赖。也不用往 Blender 自带 Python 中安装 MCP 服务。

## 2. 安装基础软件

Git、Node.js 从上表官方入口安装，按新机操作系统和处理器选择安装包。安装后重新打开终端，确认：

```sh
git --version
node --version
```

Node 主版本应不小于 22。旧机测试过 25.9.0，不要求复制这个旧版本。

### 安装 uv 和独立 Python

以下采用 uv 官方独立安装器的固定版本入口。它会执行下载的安装脚本；也可先从 [uv 官方安装页](https://docs.astral.sh/uv/getting-started/installation/) 取得并检查安装器或对应平台二进制文件。只需选新机对应的一组命令。

macOS / Linux：

```sh
curl -LsSf https://astral.sh/uv/0.13.5/install.sh | sh
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.13.5/install.ps1 | iex"
```

若电脑的组织策略阻止安装脚本，使用允许的官方安装渠道，不修改组织策略。安装后重新打开终端，再执行：

```sh
uv --version
uv python install 3.12
uv python find 3.12
```

保存最后一条命令输出的 **Python 可执行文件绝对路径**，第 3 节要填入 `pythonExecutable`。uv 管理的 Python 不一定提供名为 `python` 或 `python3` 的全局命令，因此本项目直接填写路径。Blender 安装目录内的 Python 不替代这里的独立环境。

如果新机已经有可用的 Python 3.10 以上，可以使用其实际可执行文件；安装 MCP 时把示例中的 `--python 3.12` 换成该解释器的绝对路径。首次 Python、MCP 及依赖下载需要网络。搬运 `tools/setup/` 不会省略所有这些下载。

## 3. 取得代码、资产与本机路径配置

先按 [迁移步骤第 3 节](MIGRATION.md#3-新电脑先准备基础软件并取得代码) clone `main` 并比对提交号，然后立即填写下面的 Python 路径。完成路径配置后，再按迁移文档第 4、5 节复制 `tools/`、导入并校验角色资产。所有后续项目命令都在新电脑的 `Blender` 工作区根目录执行。

将 `config/local.example.json` 复制为 `config/local.json`，修改副本。已有本机配置时编辑对应字段，不覆盖整份配置。以下是路径格式示例，**必须换成新电脑实际输出的路径**。

macOS 示例：

```json
{
  "blenderExecutable": "/Applications/Blender.app/Contents/MacOS/Blender",
  "pythonExecutable": "/Users/NEW_USER/实际Python目录/bin/python3.12",
  "charakuruConfig": ""
}
```

Windows 示例，JSON 中使用正斜杠可以避免反斜杠转义问题：

```json
{
  "blenderExecutable": "C:/Program Files/Blender Foundation/Blender 5.2/blender.exe",
  "pythonExecutable": "C:/Users/NEW_USER/实际Python目录/python.exe",
  "charakuruConfig": ""
}
```

如果 Blender 此时尚未安装，`blenderExecutable` 暂留空字符串，第 4 节安装后补填。如果 Blender 装在其他位置，填写对应位置；Linux 填实际 `blender` 可执行文件的绝对路径。`charakuruConfig` 通常保持空字符串，使用套件随附的素体。确有自定义素体等设置时，再按固定工具包的 `local-config.example.json` 创建新的 `config/charakuru.local.json`，改成新机有效路径。

`config/local.json`、`config/charakuru.local.json` 已被 Git 忽略。本页不要求复制旧电脑的绝对路径、`.codex` 配置或登录信息。

同一工作区同时从 WSL 和 Windows 使用时，可以保留 `config/local.json` 的共享配置，并在 `config/local.win32.json` 填 Windows 的 `blenderExecutable` 和 `pythonExecutable`。统一入口按当前 Node 平台逐字段覆盖共享配置；Linux 对应 `config/local.linux.json`。这些本机覆盖文件也不入 Git。Windows Blender 与 MCP 应从 Windows 启动，工程传入 UNC 路径；这不代表 WSL 已能直接驱动 Windows Blender。

若 `tools/charakuru/1.1.0/` 没有一起搬运，在填写 Python 路径之后执行：

```sh
node scripts/workspace.mjs install-tool
```

该命令使用锁定的完整 Release ZIP，并检查工具文件。它不会安装 Blender、VRM 或 MCP 插件。不要用 GitHub 的 Source code ZIP 代替完整运行套件。

## 4. 安装 Blender 和 VRM Add-on

安装 Blender 5.2.3 LTS 并启动一次，在 About 中确认版本。先保留新电脑的默认配置，不导入整套旧机偏好。跨 macOS / Windows / Linux 迁移应重新安装对应平台的 Blender，而非复制旧应用程序本体。

在 Blender 中执行：

1. 打开 **Edit → Preferences → Add-ons**。
2. 在右上角下拉菜单选择 **Install from Disk**。
3. 选择 `tools/setup/VRM_Addon_for_Blender-Extension-4_10_2.zip`，**不要先解压 ZIP**。
4. 确认 **VRM format** 已安装且启用，版本为 4.10.2；保存偏好后重启 Blender，确认插件仍启用。

固定 ZIP 的扩展清单声明支持 Blender `>=4.2.0`、`<5.3.0`，覆盖本项目的 5.2.3。迁移阶段先保持该组合，不点击自动升级来替换恢复基线。

如果漏搬了 ZIP，优先从旧电脑补搬。也可从 [VRM Add-on 发布记录](https://github.com/saturday06/VRM-Addon-for-Blender/releases) 取得相同的 **4.10.2 Extension ZIP**，按锁文件的 `optionalInstallationMedia` 核对大小和 SHA-256；不要用文件名相似的另一种发行包直接替换。

三个固定安装文件都在 `toolchain.lock.json` 中有校验值；macOS 可用 `shasum -a 256 "文件路径"`，Windows 可用 `Get-FileHash "文件路径" -Algorithm SHA256` 逐一核对。`doctor` 会校验 Charakuru 运行套件，**不会替你验证插件已经安装到 Blender 中**。

## 5. 安装 Blender MCP 的两端

连接关系如下：

```text
Codex
  → 启动本机 mcp-for-blender 服务（STDIO）
  → 连接 localhost:9876
  → Blender 内已启用的 MCP for Blender 插件
  → 当前打开的 character.blend
```

包名是 **`mcp-for-blender`**。不要与其他相似名称的 MCP 项目混装。9876 是 Blender 插件的本地连接端口，**不是在 Codex 中填写的 HTTP MCP URL**。

### 5.1 主机端：安装固定 Python 包

推荐使用搬来的 wheel，在新工作区根目录执行：

```sh
uv tool install --python 3.12 "./tools/setup/mcp_for_blender-2.2.4-py3-none-any.whl"
uv tool list
uv tool dir --bin
```

若没有 wheel，安装命令改为以下一条；不需要两种方式都执行：

```sh
uv tool install --python 3.12 "mcp-for-blender==2.2.4"
```

`uv tool list` 应显示 `mcp-for-blender v2.2.4`。保存 `uv tool dir --bin` 输出的目录，其中的 `mcp-for-blender`（Windows 为 `mcp-for-blender.exe`）就是第 6 节的启动程序。这里安装的是独立工具环境，不修改项目源码或 Blender 自带 Python。

本地 wheel 仍会安装 `mcp>=1.26.0,<2`、`httpx>=0.27.0` 及间接依赖，需要联网时属于正常情况。若 uv 提示工具目录不在 PATH，可执行 `uv tool update-shell` 并重新打开终端；Codex 配置仍使用绝对路径。

### 5.2 Blender 端：安装并启用插件

1. 在 Blender 的 **Edit → Preferences → Add-ons → Install from Disk** 选择 `tools/setup/blender-mcp-2.2.4/addon.py`。
2. 搜索并启用 **MCP for Blender**（可能带有 `Interface:` 前缀），保存偏好。
3. 按第 7 节打开唯一主源。在 3D 视图按 `N`，打开 **MCP for Blender** 面板。
4. 确认端口为 **9876**。固定版实际按钮为 **Connect to MCP server**，连接后显示 **Connected on port 9876**。部分上游说明称其为 “Start MCP Server”；以本固定文件的界面为准。如果已自动启动并显示连接状态，不必重复点击。

该 `addon.py` 与保留的 2.2.4 wheel 内插件逐字节一致。插件内部显示的版本可能是 `1.8`，与 Python 包 `2.2.4` 属于不同版本标识；应以安装文件 SHA-256 和包版本核对，不能据此误判为装错版本。

如果没有保存的 `addon.py`，可从已经安装的固定包提取安装：使用第 5.1 节确定的程序路径运行 `mcp-for-blender addon-paths` 查看位置，再运行 `mcp-for-blender install-addon`。后者会安装该包自带插件，之后仍要在 Blender 中启用。无需运行会配置其他客户端的通用 `setup` 命令，也不要拉取未锁定的 `latest` 插件。

恢复期间只启动一个用于制作的 Blender 实例。插件显示端口已连接不代表 Codex 已完成 MCP 握手，还要完成下面的客户端配置和实际读取验收。

## 6. 配置 Codex 的本地 MCP

按 [官方安装入口](https://developers.openai.com/codex/app) 安装桌面应用并登录，打开新电脑上的 `Blender` 工作区。GitHub 的 clone、提交和推送认证单独配置，不复制旧机器的私钥或整套账号配置。

按 [OpenAI 官方 MCP 文档](https://developers.openai.com/codex/mcp)，可以在桌面应用 **Settings → MCP servers → Add server** 中添加本地 **STDIO** 服务，或编辑配置文件。只选择一种方式，避免重复注册相同服务。

配置文件通常位于：

- macOS / Linux：`~/.codex/config.toml`
- Windows：`%USERPROFILE%\.codex\config.toml`

如果本机自定义了配置目录，使用应用实际读取的配置。以下片段只合并到现有配置中，不覆盖其他设置；已有 `[mcp_servers.blender]` 时编辑原条目，不重复添加同名表。机器专属路径保留在用户配置中，不提交进本项目。

macOS / Linux 示例：

```toml
[mcp_servers.blender]
command = "/Users/NEW_USER/.local/bin/mcp-for-blender"
args = []
startup_timeout_sec = 60

[mcp_servers.blender.env]
BLENDER_HOST = "localhost"
BLENDER_PORT = "9876"
DISABLE_TELEMETRY = "true"
```

Windows 示例，TOML 单引号字符串保留反斜杠，无须写成双反斜杠：

```toml
[mcp_servers.blender]
command = 'C:\Users\NEW_USER\.local\bin\mcp-for-blender.exe'
args = []
startup_timeout_sec = 60

[mcp_servers.blender.env]
BLENDER_HOST = "localhost"
BLENDER_PORT = "9876"
DISABLE_TELEMETRY = "true"
```

**`command` 必须替换为第 5.1 节查到的真实路径**，不要照抄 `NEW_USER`。GUI 添加方式使用相同程序路径、空参数，以及上述三个环境变量。桌面应用不一定继承终端 PATH，所以不能仅凭终端中命令可用就省略绝对路径。

保存后按 MCP 设置页的 **Restart** 重启连接；若界面没有该入口，重启桌面应用后新建一个会话验证。MCP 服务由客户端启动，不需要另外开终端常驻运行。手动执行它后静默等待读取 STDIO，不代表程序卡死。

只保留本机回环地址。无需云端 MCP 地址、API Key、Tripo Token，或把 9876 暴露给局域网。`DISABLE_TELEMETRY=true` 沿用本项目的设置。

## 7. 按顺序验收，全部通过后续作

### 文件层

资产复制后、任何建模保存之前执行：

```sh
node scripts/assets.mjs verify --project 6c8500bf-1c38-481a-babd-e5478a50dd67
node scripts/workspace.mjs doctor
node scripts/workspace.mjs validate-source
```

当前基线应有 963 个资产文件、零不一致、零额外资产文件；Charakuru 应有 277 个文件校验通过。**先 verify，不能先 inventory 重写基准。** `doctor` 不测试 Blender/MCP，`validate-source` 只检查语法，不会执行模型修补脚本。

### Blender 和 MCP 层

用 Blender 打开：

```text
projects/6c8500bf-1c38-481a-babd-e5478a50dd67/work/character.blend
```

这是建模主源。不要从 VRM 重新导入代替它，也不要重跑已经执行的 `finish_comprehensive_refinement.py`。

确认无丢失贴图提示，材质、骨架、形态键和马尾物理设置仍在；固定版 VRM 插件已启用。原交接报告记录主源有 19 张文件图片内嵌、128 根骨骼、零外部链接库，可作为恢复比对线索，仍需在新机实际读取确认。

在 Codex 新会话中提出只读验收：

> 先不要修改或保存模型。通过 Blender MCP 的 get_scene_info（fields 包含 settings）读取当前场景，报告实际打开的 Blend 完整路径、Blender 版本、主要对象和骨架情况；核对它是本工作区的 work/character.blend，并确认 VRM 插件可用。不要只报告服务已启动。

如工具被延迟加载，由代理搜索 Blender MCP 场景工具后调用；必要时再用只读查询检查插件与数据。只有实际调用返回正确工程，才记为 MCP 验收成功。如果本机安装了 Codex CLI，可用 `codex mcp list` 辅助检查配置，但列表出现服务名称不能代替场景读取。

### 应用与预览层

按需分别启动，命令保持运行时在浏览器打开打印的地址：

```sh
node scripts/workspace.mjs start
node scripts/workspace.mjs preview
```

两条长驻命令需要不同终端；只想查看角色时先启动 `preview` 即可。

| 服务 | 默认地址或端口 | 检查内容 |
| --- | --- | --- |
| Charakuru 应用 | `http://127.0.0.1:8768` | 从复制的 `.charakit` 恢复正确角色，不恢复旧的排队请求 |
| 独立审阅页面 | `http://127.0.0.1:8769` 下命令打印的完整路径 | 当前静态模型、眨眼、口型和短动作 |
| Blender 插件 | `localhost:9876` | 供主机 MCP 服务连接，不是网页 |

`.charakit` 中保留了较早的流程阶段，这不表示模型丢失，不要重新触发 Tripo 生成。短动作检查通过后，记录新机软件版本、主源身份和发现的问题，再使用 [续作提示词](CONTINUE_PROMPT.txt)。旧电脑原件和传输副本保留到恢复成功。

安装成功和上述恢复检查不等于模型已通过外观、性能或客户验收。当前仍是审阅版，已知问题见 [交接说明](HANDOFF.md)。

## 8. 常见问题

| 现象 | 检查与处理 |
| --- | --- |
| clone 后没有模型或贴图 | Git 不含大资产；按迁移文档导入整个角色目录并验证 |
| 提示找不到 Python | 将 `uv python find 3.12` 输出的真实路径填入 `config/local.json`；路径值不含命令参数 |
| MCP 启动时下载很久 | 先在终端完成第 5.1 节的工具安装，处理网络或依赖报错，再重启 MCP；不要启动重建模型来测试 |
| MCP 程序不存在 / spawn ENOENT | 检查 `uv tool dir --bin`、Windows 的 `.exe` 后缀和 `command` 绝对路径，重启桌面应用 |
| Connection refused / 超时 | 确认 Blender 已启动、插件启用、面板端口与 `BLENDER_PORT` 相同，并排查多 Blender 实例抢占 9876 |
| MCP 可以读取但工程不对 | 比较返回的完整 Blend 路径，切换至正确主源并重新读取；不向错误场景写入 |
| Windows、WSL 的路径或 localhost 混用 | 本指南采用 Windows 原生 Blender + Windows 原生 uv/Python/MCP；先保持同一系统环境，WSL 不作为恢复必需项 |
| VRM/MCP 插件未出现 | 核对所选文件及安装位置；VRM 安装 ZIP 不解压，MCP 安装 `addon.py`；启用后重启 Blender 再查 |
| 8768 或 8769 被占用 | 不结束无关进程；使用 `start --port 8778` 或 `preview --port 8779`，打开新打印地址 |
| 复制或哈希校验不通过 | 保留两端原件，补齐缺失/损坏文件并再次 verify，不用 inventory 把错误副本变成基准 |

## 参考依据

官方页面读取与本地安装介质核对日期：**2026-10-28**。软件安装页面会变化；项目版本仍以 `toolchain.lock.json` 为准。

- [OpenAI MCP 配置与 STDIO 说明](https://developers.openai.com/codex/mcp)：配置位置、`command` / `args` / `env`、桌面设置与重启；该入口当前重定向至 `learn.chatgpt.com` 官方文档。
- [OpenAI 桌面应用安装说明](https://developers.openai.com/codex/app)：按平台安装、登录和打开本地文件夹；官方入口可能使用 ChatGPT 桌面应用名称。
- [uv 安装](https://docs.astral.sh/uv/getting-started/installation/)、[Python 管理](https://docs.astral.sh/uv/guides/install-python/)、[工具隔离安装](https://docs.astral.sh/uv/guides/tools/)。
- [Blender LTS](https://www.blender.org/download/lts/)、[VRM Add-on 安装](https://vrm-addon-for-blender.info/en-us/installation/)。
- [MCP for Blender 上游](https://github.com/ahujasid/blender-mcp)。本页的 2.2.4 行为以保留 wheel 中的包元数据、`server.py`、`telemetry.py` 和 `bundled/addon.py` 为依据；没有用上游 `main` 的最新代码替换固定版本。
- 本地安装介质的来源与 SHA-256：[`toolchain.lock.json`](../toolchain.lock.json) 的 `optionalInstallationMedia`。它们已在旧机核对，不代表已经在新机安装成功。
