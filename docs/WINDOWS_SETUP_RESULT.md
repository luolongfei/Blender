# Windows 工具链安装与读取验收

验收日期：2026-10-28（America/Los_Angeles）。资产恢复沿用 `state/migration-restore.json`，本次没有重新解压角色资产、运行制作脚本或保存模型。

## 安装结果

- Windows 原生 Blender **5.2.3 LTS**（构建 `e13b728c6880`）。官方 MSI 哈希与签名通过，但因 Program Files 写入权限不足而回滚（1603 / 1303）；改用官方同版本 x64 ZIP，SHA-256 与官方下载目录校验文件一致。程序安装在 Windows 当前用户的 LocalAppData/Programs/Blender Foundation 下。
- VRM Add-on **4.10.2** 从锁定 Extension ZIP 安装，启用模块 `bl_ext.user_default.vrm`。
- **mcp-for-blender 2.2.4** 从锁定 wheel 安装到 Windows uv 独立工具环境，使用 Windows Python **3.12.15**。现有 uv **0.12.22** 保留，未强行升级到旧机记录的版本。
- Blender 端 MCP 从锁定 `addon.py` 安装，安装后 SHA-256 仍为 `8e193a249e2ed1262f3e794ee2254efd8e3d1f0224385245f2b989154c837083`。偏好已保存，关闭安装进程再启动主工程后，两个插件仍启用。
- Codex 用户配置已备份并添加 `blender` STDIO 条目，使用 Windows 绝对程序路径、`BLENDER_HOST=localhost`、`BLENDER_PORT=9876`、`DISABLE_TELEMETRY=true`、60 秒启动超时。实际监听地址为 **127.0.0.1:9876**。

本机具体程序路径见 `config/local.win32.json` 和 `state/windows-blender-process.json`。WSL 的原 `config/local.json` 保留；统一工作区入口增加按平台覆盖配置，未改 Charakuru 上游工具文件。

## 实际读取结果

使用 Windows MCP Python SDK，从上述 Codex 条目读取启动配置，完成 STDIO initialize、tools/list，并实际调用 `get_scene_info(fields=["settings","details","materials","modifiers"])` 和只读 `execute_blender_code`。原始响应保存在 `state/windows-mcp-verification.json`。

- 打开文件：工作区 UNC 路径下 `projects/6c8500bf-1c38-481a-babd-e5478a50dd67/work/character.blend`，与唯一主源一致。
- 场景：`双马尾绿裙 · Local Atelier`，277 个对象（251 个网格、1 个骨架、7 个相机、15 个空对象、3 个灯光），36 个材质。
- `Character_Rig` 有 **128 根骨骼**；VRM 扩展数据可读取，14 组弹簧骨、8 个碰撞体。
- 20 个网格具有形态键，其中脸部对象有 24 个形态键（包括 Basis）。
- **19 张文件图片全部内嵌**，读取图片尺寸与像素缓冲后均成功解码；**0 个外部链接库**。
- Windows Blender 窗口实际显示主工程，标题版本为 5.2.3 LTS。
- 打开前后主源 SHA-256 均为 `6e7361e7f5c306325d3f33c4c9f6f56b1a489d5f4579bb4be951d4c3d59cc8c1`，未覆盖主源。

MCP 握手中的 serverInfo.version 为依赖库版本 1.30.0；包版本由 uv 确认为 2.2.4，二者不是同一版本标识。具体依赖记录在 `state/windows-mcp-dependencies.txt`。

## 范围与继续使用

Windows Blender、插件和 MCP 协议调用验收通过。当前聊天启动时的工具目录未包含新服务；本次实际场景读取由 SDK 按 Codex 已保存配置发起，**不声称当前聊天已经热加载原生 Blender 工具**。后续在桌面应用重新加载 MCP 连接，或重启应用后开启会话，首先调用场景工具核对完整主源路径。

本次没有执行重渲染、全量碰撞检查、重新导出或模型修改；物理与形态键的读取不等于完整动作测试。模型继续保持审阅阶段，外观和客户验收状态不变。

参考：[Blender 官方版本目录](https://download.blender.org/release/Blender5.2/)、[OpenAI 官方 MCP 配置说明](https://learn.chatgpt.com/docs/extend/mcp)。
