# 已接收的制作交接

## Git 收录整理与源码检查（2026-10-29）

本轮保存当前重建、独立动作预览和历史制作的源码、网页、说明及小型证据。根目录迁移 ZIP、模型、贴图、渲染、缓存、工具安装文件、本机配置和预览服务运行状态继续留在本地，不纳入 Git；未删除或覆盖大资产。

源码检查改用标准输入传递 Python 文件清单，解决 Windows 命令行长度限制，并覆盖 Character_Rebuild 及独立预览源码。实际检查：768 个 Python、49 个 JavaScript 文件语法通过；12 项测试中 11 项通过、1 项因 Windows 符号链接权限跳过，包含超长清单和语法错误检出回归测试。主源、静态 Blend/GLB、动作预览 Blend/GLB 共五个关键文件重新计算 SHA-256，均与登记相符。

角色顶层状态、续作提示、旧预览标识已区分当前重建与 R09/R11 历史。旧迁移清单保持历史身份；本轮不是新的资产迁移或冻结备份，也未重新进行模型外观、动画或全局碰撞验收。提交与远端同步状态以 Git 为准。

## 新增：本机网页动作预览（2026-10-29）

用户明确要求在网页预览、拖动和播放动作。派生预览位于 `Character_Rebuild/interactive/`：`rigged-preview.blend` 与 `animated-preview.glb`，独立于静态交付和正式主源。预览源SHA `fee63d90b281373b56e453af4e0402126aef635105a17bca88f7893e8e073738`；GLB SHA `64febee7ea3285cc8f4ff91725fbc2df686daaaaf50e2508384abb8ddec40289`。

47模型对象、22骨骼、5内嵌贴图，idle/wave/look/walk与Tpose共5clips。网页提供4个实际动作及单独T形姿势还原；walk准确标为原地轻踏。GLB在干净Blender场景重导后，静止几何对E8最大点距1.85e-7，动作同相位包围盒差4.17e-7；权重归一和真实动画轨道通过。Blender自动产生的Icosphere骨骼显示辅助按实际custom_shape引用单独排除，不计模型。完整记录见 `interactive/work/export-reimport-report.json`、`foot-contact-report.json` 及 `interactive/reports/`。

网页已实际验证载入、拖动旋转、滚轮缩放、动作切换、暂停、时间轴、半速播放、头肩/侧面/斜前镜头与Tpose恢复。浏览器实拍为 `interactive/checks/browser-wave.jpg`。启动 `node Character_Rebuild/scripts/serve_delivery.mjs --port 52860` 后访问 `http://127.0.0.1:52860/interactive/viewer.html`；服务只绑定本机127.0.0.1，使用锁定工具的本地Three.js依赖。旧图片检查页仍在 `/review.html`。

限制：有限幅度基础蒙皮预览；头发随头部、裙子随骨盆，没有布料/头发物理、独立手指或面部表情。斜侧面腿、袜、鞋的纹理串色为原E8已有问题，绑定未改变纹理。没有进行动态全局碰撞认证，不代表生产绑定或客户外观验收。原静态交付、正式主源及旧资产未覆盖；旧VRM仍是历史资产。

## 当前：Hunyuan静态动漫重建交付

用户随后明确要求改用根目录新GLB重建，不继续旧R11模型。本轮交付目录为 `Character_Rebuild/`，其 `character-rebuilt.blend` 与 `projects/6c8500bf-1c38-481a-babd-e5478a50dd67/work/character.blend` 字节一致。原主源已备份于 `work/stage-backups/character-before-hunyuan-3d0e397a6ba0b874.blend` 并校验。

GLB：`4b70a67d9f2677b5d80e7d6d79a31fd6ff90d6d2c6c267c9a50ba678f3c904d8`；可编辑源：`b3506fb27cdc04633a5bdaddf5e14166059209bab0f115e790614a3b6bd3d4f5`。已实际在空Blender场景重导入，全部47个网格逐个内部接触检查为0；文件包含238527三角、16材质、5张内嵌贴图，无骨骼/动画。具体范围与限制见 `Character_Rebuild/完成说明.txt` 和对应最终报告。外观仍需艺术精修，不能视为原画完全还原或客户认可；未验证跨部件全局碰撞及动画。

旧R11、旧VRM、旧.charakit应用状态、旧缓存和迁移清单保留作历史，不得据其覆盖这份新工作主源。以下停止记录仍适用于旧R11路线；本轮新GLB是用户另行授权的新任务。没有异机备份声明。


## 历史记录阅读说明

以下均为各阶段当时的记录。文中的“当前”“正在运行”“下一步”及主源哈希仅属于该历史阶段，不代表当前状态或继续制作指令。R11 已停止；当前主源与独立动作预览以本文顶部及 project-state.json 顶层字段为准。旧资产清单不覆盖新成果，也未完成当前资产的异机备份。

## 历史记录：用户已要求停止并交付现状（2026-10-29）

用户明确表示对效果不满意，要求立即停止制作、直接交付。目标状态已设为 paused；不得因下方历史待办自行恢复制作。各制作分支已停止，无新模型修改。当前最后成功保存并登记的独立源是 `projects/6c8500bf-1c38-481a-babd-e5478a50dd67/work/reconstruction-r02/reference-adult-fitted-bands-r11.blend`，SHA `8fd87255aa7576eeaa486cad396773350b608b27b5e57914332165afa67eb2d9`。正式 `work/character.blend` 和既有 `exports/avatar.blend` / VRM 保留原阶段，未覆盖。

现状交接目录为该项目 `exports/stopped-handoff-2026-10-29/`，打包路径 `exports/stopped-handoff-2026-10-29.zip`；是否完成须查看同名 `.verification.json`，不能仅凭这条说明宣称校验完成。包内 `current-editable.blend` 应与8fd8逐字节一致，17张图像已内嵌、没有链接库；301个旧衣物缓存文件按相对目录保留，但旧缓存不是新模型动态通过证据。原始、失败与待合并副本均留在原工程，本地打包不等于异机备份。

停止时尚未合入：新放松嘴角、新上睫、连续外眼角诊断、冠部毛流/14根耳侧末梢、鼻部方向和新腿足鞋袜。口角/睫毛已看图选择了方向，但没有保存为模型；外眼角单图在停止前已自然结束且未保存/未裁决；冠部实际失败未保存。最后额外离线鼻/冠末梢候选也没有安装。不要把这些预览、计划、旧VRM或旧检查报告当作当前交付已完成。

此交付只移交已有成果，外观未获用户认可，未达到目标质量；相关限制与文件对应关系以包内 `交付说明.txt`、`project-state-at-stop.json` 和 `sha256-manifest.json` 为准。

## 历史记录：R11 本轮实际诊断与待执行队列（当前仍0df7）

上睫 `upper-lash-neutral-preview-r11/before.png` / `after.png` 已由整合者实看，贴根、前伸轻卷的方向选中，原生中性根/毛干越限0；没有保存源，finally报 Original pose / external scene differs，不能记完整通过。选型报告 `upper-lash-neutral-visual-selection-r11.json`。正式新睫仍待最终眼形后重新绑定，旧预览映射不能直接复用。

颏喉侧面补图v2仅完成 `chin-throat-contour-side-completion-r11/side-0.png`，整合者已实看；source29e3，头脸实际pre/post逐点exact、源文件exact，但渲染后24根手/指骨出现67个author通道叶子变化，最大5.960464477539063e-8，5个外部对象positionHash变化。非纯derived缓存差，完整记录在同名side-0.json；side-1因第一进程失败尚未运行。hand v6实际恢复矩阵误差同量级，代码用matrix_basis恢复而未精确保留原通道；Action重新求值机制尚待 `rig_frame_one_channel_diagnostic_r11.py`（ed261a52…）核对。不得原样重跑渲染/全外部几何来猜原因。

当前嘴角整脸单图8ee286cb…正在0df7上运行：实际20111依赖点exact、中央封唇/眼耳/低颈保护exact，原521局部交叉变0、最大2.484216mm；还没看图、未存源。下一保存队列：环带6579f548…（proof1c4678b4…）→冠部3bd5aae3…；随后完整quarter/front。连续新眼缘单图d8e3c6f9…可排，明确仅艺术诊断、保留两已知桥区交叉，不改严格安装门槛/不保存。不要把这些未执行入口算模型进步。

MPFB最小CC0表情资源已取齐约2.14MB，位于 `assets/mpfb-facial-pilot-r11/`，有固定commit/许可/哈希；离线6环域86条初步对应在 `work/reconstruction-r02/mpfb-face-pilot-offline-r11/`。未直接套到当前头，不能当完成表情。鞋袜腿组合0ae1caa9…与六态45b06a7b…仍待root串行实际检查。唇微纹理诊断继续后置，先解决整体眼口鼻造型。

## 历史记录：R11 当前 0df7：双马尾完整两图已看，当前发型与默认脸仍待改善

`reference-ties-v3-r11/full-groom-front.png` 与 `full-groom-quarter.png` 已完成并由整合者实际查看。扎位更接近耳后低位，可继续；头顶前后毛流露缝、悬空发圈、过厚且向前外侧硬弯的尾束仍明显，不能称发型完成。原画低双马尾是保护设计，失败旧3D尾形不是不可变约束；先环带/冠部，之后重做自然受重力的尾形并重验依赖。

479a 唯一 post-tie 观察已从0df7成功运行，`work/reconstruction-r02/reference-post-tie-actual-r11.json/.npz` 共179数组，未保存Blend。hair代理据此离线校准，环带→冠部安装尚未执行。上睫neutral-only同机位两图正在串行执行；颏喉side补图随后。MPFB资源研究已保存 `facial-expression-asset-assessment-r11.json`，明确不存在已验证可直接套当前头的AAA表情库；任何迁移仍需当前拓扑适配与实际验证。

最新独立源 `work/reconstruction-r02/reference-adult-reference-ties-v3-r11.blend`，SHA `0df7249b6d9ec4e3e3021d606dfd006d0baf5d9eb81b406bb1b86ccb50a4917e`，接下方29e3。实际v3当前表面与native检查errors=[]；25条局部修弯、5条真实旧入肤p1/p2例外、10296/10301两条小于14mm幅度例外全部记录。发根p0、全部尾pin/末3、46脸侧发与其余目标保护。当前technicalAcceptance只属本阶段原门槛；仍有容差内负净距，不能据此宣称全局无穿插或动态通过。正用v10e相同完整机位渲染front/quarter，输出 `reference-ties-v3-r11/`。下一唯一采样为 `observe_reference_post_tie_complete_r11.py`（479a76f0…），取成功v3源；冠部与发圈方案共同消费它，尚未运行。

29e3 warm0/.2四张整脸已看，轻微笑意不足以解决木然，默认warm仍0。新三视艺术参考 `references/generated/natural-face-profile-reference-r11.png`（80d2dd47…）及同名完整提示/来源JSON已保存；只是生成的近似造型方向，不是3D成果/标定扫描，也未确认工具实际模型版本。新真实head-neck只读v2成功（54数组、恢复exact，NPZ714ccfb1…），当前头净角.01079度、chin到前领口约70.485mm，不能把视觉长直颈简单解释为颈长度超标。耳高约51mm、球横径24.59mm、眼裂宽23.68mm，暂不放大球/缩耳；face候选优先改善颏下→前喉转折。

颏喉内存preview已完成quarter-1和真实几何诊断，但渲染后原pose合同检查失败，未生成侧面/未保存源。复核不能证明是新camera/Key，因为合同仅含原ARMATURE；独立 `chin_throat_contour_preview_r11_v2.py`（0a2bdd12…）准备好，以原29e3分两个fresh进程仅补side-0/side-1，保存author/derived/actual geometry三层精确差异；不凭容差猜原因。当前quarter略有改善，尚不采用。

外眼角两neutral诊断图也已看：新宽场消除了旧大折板/纵沟，却仍有眼裂台阶/S反曲，未采用、未写源。blink正在从实际皮肤桥终端和外半眼缘构建连续目标；旧radial Key仍0，完整眨眼/inner链暂缓。新上睫另准备neutral-only两图 `adult_upper_lash_neutral_preview_r11.py`（fa2f8a24…），当前场景一次捕获，不用旧f501数据；正式v4动态验证仍等最终眼形。唇纹原/5微米诊断入口6439d94a…已备未跑。正式主源与导出继续保持旧基线，当前不具备最终交付条件。

## 历史记录：R11 当前 29e3：自然闭口已保存并看完整脸，整体默认神态尚未定稿

最新独立续作源 `work/reconstruction-r02/reference-adult-relaxed-lip-seal-r11.blend`，SHA `29e3e6b59a08487643d28642975a6ab1f92ca4af830f484af291f25434e08df8`。从下述5f30接续；实际五个seal/warm组合检查完成，旧口角真实穿入明确保留。`natural-rest-overall-r11/front.png` 与 `quarter.png` 已由整合者实看：唇缝缩细可保留为局部进步，但全脸仍木、嘴角下垂、眼区显弱、颈部显长直，绝非整体外观通过。正在拍warm=.2的两张轻微笑意对照；没有据此更改源默认warm0。face代理转入真实眼裂/面宽/耳/下脸比例和REST/POSE头颈矩阵核对，不继续盲叠细部掩盖大形。

原画双马尾扎位v2从29e3实际运行失败，未保存：真实head表面control中点32、native点55、中点65低于-.5mm，另2处真实跨面探针超限。`reference-tie-reconnection-v2-r11.log`及attempt/arrays保留，hair代理用此次实际数据局部修正；不得放宽阈值或跳过原生曲线。唯一post-tie采样应使用 `observe_reference_post_tie_complete_r11.py`（479a76f0…），尚未运行。

眼角下一步只跑两张neutral诊断（旧场低强度/宽缓表面场），不保存源、不跑完整新九态；整体眼眶方向成立前暂停midpath/inner。最新绑定准备为binder_v7（ce0f3498…）与prearm_v4（5c22eaf1…），新睫v4（3b573fc7…）补早段三态，均未实际安装。足部225mm候选已离线放置，待独立腿片补采；衣物接触代理也仅完成离线准备。主源与正式导出仍未更新。

### 上一阶段 5f30：眼角候选拒绝并复位

最新独立源 `work/reconstruction-r02/reference-adult-outer-eye-review-reset-r11.blend`，SHA `5f30122cf574029b53682d9d690b94aea5e3e12f343fc8b5b63bacca6d8a1d72`。链为下方3d14新手 → radial外眼角bc96db15 → 当前5f30。bc96实际9态技术检查完成，但 `outer-eye-radial-eyes-r11-v2/frame-0001.png` / `frame-0002.png` 同机位图显示新外上睑折板和外侧纵向沟，整合者与两个代理实看后拒绝强度1；5f30仅将 `R11 refined outer eye corners` Key设0，其余Key精确保留，原可编辑候选未删除。旧眼角仍偏钝，不能称外观通过。

闭口入口 `relaxed_lip_seal_install_r11_v2.py` 已从5f30成功保存上方29e3；整体图优先于嘴部近景完成。旧lower对象属性始终0，radial Key0，静态修形不能当表情清空。眼角全多态观察仍暂缓，不能沿用旧source证明后续新组合。

## 历史记录：R11 当前 3d14：新手腕连续表面检查通过，默认自然闭口正在整合

当前独立源 `work/reconstruction-r02/reference-adult-refined-hands-r11-v6.blend`，SHA `3d14a1adc5217d16d74a8aa3ace68547b773fe8b4b93485de7127344bf3e11fc`，接下方31df。实际静置、屈腕、伸腕两侧的连续三角面交叉/数值接触均0；不是只检查顶点。旧v5右腕34/16/46对真实交叉已用L1真实传递定位并修改19个原始支撑点，v6成功报告为 `refined-hand-integrate-r11-v6.log`。手部近景尚未看，不等于外观通过。后续源始终核对project-state最新候选与SHA。

31df眼睑roll与唇颏v2的同机位0/1图已实际查看，均保留为局部改进；旧唇下v1亮横纹继续默认0。整体quarter仍显示钝眼角、直立旧睫毛、真实开唇黑带及冠部前后毛流分离露头皮，不能宣布面部外观成立。冠部实际47对相邻发根在前5mm方向分叉，正在独立修毛流；原图低黑双马尾与46条误归gather的脸侧发必须保留。扎位v2与环带收紧尚未实际完成。

用户最新默认表情要求已记录：放松轻闭口、自然眼神，轻笑可调，不固定大笑。新资源 `references/generated/natural-default-expression-reference-r11.png`（f1e9a592…）及同名JSON为生成的艺术参考，不是3D成果；工具未暴露模型选择，不能宣称确认使用了image2.5。`relaxed_lip_seal_install_r11_v2.py`已离线冻结，实际安装待排；旧口角残留穿入明确未解决。新jaw_v2将闭口连接纳入同一次保存事务，旧upper-oral-clearance暂停自动安装以免破坏新唇接触。

外眼角radial首轮因Blender5.2取消Raycast.mapping失败且未保存；独立v2仅改Interpolation菜单并实际读回，当前正从3d14运行。新眼形确定后必须重新观察并重建眨眼中段/内衬，随后新上睫v3；不得用旧f501观察证明新组合。足部实际capture已从31df完成，225mm等只是艺术适配候选，不是人体标准。主源3d0e397a、正式导出及最终交付状态均未改变。

## 历史记录：R11 当前 31df：原画前发与上睑减厚已保存，眼口组合实图正在审阅

当前独立续作源 `work/reconstruction-r02/reference-adult-lower-face-continuity-v2-r11.blend`，SHA `31df467c65168468b2a246e169e165a3f3e870bcb0eccabac8d0377c565c8b4b`；已登记。实际链：下方284d复位 → v10e前发8dea7f21 → 上睑roll bd8e02b1 → 当前唇颏v2。后续始终以 `project-state.json` 的 `reconstructionR11.latestIndependentCandidate` 与实际SHA核对，不凭下方历史“当前”字样取旧源。

v10e实际26046根评估发丝通过，54根旧误报已以当前源真实导线88/94插值证明，23314根非影响发丝精确保留。`adult-full-groom-v10e-r11/full-groom-front.png` 已渲染并由整合者/发型代理查看：左太阳穴硬刺明显消失，右主露额保留；中央两尖厚束仍偏硬，非最终外观。原黑色低双马尾保留。实际扎位观察v2已从8dea捕获，正按原图做整束/全部pin/双环共同运输与gather连接；现扎点偏低外、环松悬，不能视为设计保护。尚无新扎位源。

bd8e上睑roll中性最大后收0.638652mm，blink .2/warm early为0.413837mm，.5/.8/.9/1及warm late全头与眉睫精确不变，实际8态已通过。它仅减薄外上睑，未包含外眼角修形。旧外端投影方案已因新增交叉/面积失真拒绝，角向连续方案仍在离线试验，未安装。旧midpath不得先装；最终新眼形选定后重采8态并重建依赖，眨眼外端需单调收合，不能保护旧half钝角造成早期反张。

31df唇颏v2采用整段连续母线，保留旧v1层且值0；公开Key `R11 lower face contour v2` 默认1。实际0/.5/1 warm检查通过，最大位移1.457151mm。眼0/1与mouth quarter0/1将从同一31df源实际渲染；外观尚未裁决。中性嘴前方约1.175mm开口为真实内唇可见，不是丢面，v2没有解决这条黑带，仍是后续独立问题。

头颈尚未绑定，最新独立入口为 `adult_head_skin_bind_r11_v5.py`（46ce29aa…）/`adult_prearm_eye_space_r11_v3.py`（0f9861e4…）。实际MCP临时节点证明Transform Geometry菜单只能用`Matrix`，不是`MATRIX`；节点已删除、交互旧源未保存。晚段上睫v3准备完毕但须等新眼形+midpath后的新观察；它使用独立组合validator，下游不能覆盖旧公开helper来绕过冻结依赖。手v5保持v4形状，新增连续面穿插验证，尚未实际运行；旧v4仅顶点包含不够证明表面安全。正式主源仍3d0e397a，导出、外观验收及最终动态交付均未完成。

## 历史记录：R11 当前 284d：唇下候选实图拒绝并复位，继续眼形与原画发型

当前独立续作源为 `work/reconstruction-r02/reference-adult-lower-face-review-reset-r11.blend`，SHA `284d4151b03232ccaa3e4ba81005f7c7bbf5c95fbc0740713200693d0947f778`。链为下方 f501 → 自然肤色 9257f2e5 → 唇下候选 c4dc → 当前 284d。实际同机位 quarter 0/1 两图已看，strength1 出现新唇根亮横条，整合者与造型代理均否决；284d 仅将该可编辑层默认值设回0，原c4dc和全部字段保留。已登记实际来源与图像报告。主源及正式输出未更新；失败候选不可代替当前源。

实际 `adult-combined-portrait-r11-v1/combined-portrait-front.png` 已查看：自然肤色可保留继续制作，仍有厚浅色眼睑环、嘴下陡坡、中央齐尾发束和左侧硬刺等外观问题。该图同时使用临时均衡灯光与 warm=.6，不能把变化全部归于材质。c4dc 技术检查通过不等于造型通过，下一唇颏候选须重做连续曲面。原画黑色低双马尾与具体角色五官是设计依据，不以生成参考或泛化族裔模板替换。

实际 v10d 原生中央发丝路径无超限，但54根同一gather组的连续插值影响被旧矩形域拒绝，未保存；准备以实际两导线插值证据判定的v10e，不任意扩大白名单。手腕v4仍失败未保存，右侧三姿态均无外露、最小正净距0.0812–0.1307mm；0.15mm只是原先自选工程余量，不是人体物理或平台标准，下一步查真实L1传递及连续接触，不盲目缩形追数值。

眼形路线调整为先修中性外上睑厚圈/外端收束，保留.5及以后旧实际状态，再重新采样7态、重建midpath/inner依赖，最后重建新上睫根线与晚段修正。不能先装旧midpath再沿用已改变neutral的旧ROI证明。新控制优先使用现有Key系统；binder最新独立v4已精确适配原lower-continuity标量。

上睫 v2 在实际中性与半闭检查通过，但全闭毛干穿入皮肤约 0.69/0.73 mm，已拒绝保存，不能采用。最新手腕 v3 只读探针完成，左右三姿态均无外露；右侧最小间距 0.0498–0.0993 mm，未达到 0.15 mm，正做右侧局部修正，左侧保持。新眨眼内衬必须在上睫完成后安装并重新检查组合；反序需重新观察睫毛，不能沿用旧眼区观察。

头颈 binder_v3 对新下脸标量驱动有精确白名单遗漏，需独立兼容版，不可放宽所有 head 引用。旧 R09 布料入口固定旧身体碰撞副本和动作，不适用于新腕/绑定；新动态须从当前实际表面建立接触体。网页 GLB 预览没有骨架与连续动画，不作为最终 VRM、绑定或表演证明。

## 历史记录：R11 当前f501fc1f：局部原画刘海v9c已保存，完整正面图已查看

当前独立续作源为 `work/reconstruction-r02/reference-adult-front-groom-v9c-r11.blend`，SHA `f501fc1f4c1004bd265fb3df20c63029f6a003764fedf5f0c0528bcf7bcf93e4`。链为cf8 v8毛发 → 758532完全闭眼修复 → 41ce34外上睑修形 → 9497a2眉半径 → 当前局部刘海v9c。下列较旧“最新”只作历史。

758532的两张实际闭眼/笑眨眼近景 `blink-corner-focused-r11/` 已完成并由整合者查看：full无露球裂缝，.5开口上缘偏平、眼角偏方，.9外侧浅灰条需分清球隙或睫毛高光。full真实露球采样0、局部新三角化无自交，中性头面与6组眉睫坐标精确不变；中段原有径向负净距未宣称解决。41ce外上睑最大收0.30mm，眨眼.4为0.15mm，.8/.9/1全头坐标精确不变；新外睑局部画面仍待看。

`brow-lash-coverage-r11/` 两张真实图已完成并查看，仅采用眉半径1.5倍，保留旧data与旧绑定合同。睫毛加粗未改善发白，明确不采用。`neutral-lighting-ab-r11-v2-front/` 两张真实图已完成：原光过平、新光过硬且头发层次不足，未采用新灯光；仍显出下巴凸球/唇下沟及皮肤蜡感，后续需实际修形和材质。当前f501仍用原灯光。

v9b失败真实探针已确认是POSE/REST之间18个骨父对象matrixLocal不同，恢复源姿态差异为空；v9c同REST前后和恢复源姿态两组严格合同均通过。98条局部guide修改，19767根非目标最终发丝逐点保持，全部26046根最终发丝检查通过。完整正面图 `adult-full-groom-v9c-r11/` 已完成且整合者/发型代理均实际查看：右侧主开口改善，但中央厚直黑帘/末梢齐平和左太阳穴硬刺仍不合格。左硬刺已据真实NPZ定位为未改gather role41低根向上回折约21.9mm，不能再只降role10刘海lift。原画低黑双马尾仍为唯一造型基准；另只读核对扎点相对耳垂，未移动尾发。预览图已送Codex面板（工具返回queued），不是最终交付图。

手部细化源 `reference-hand-nail-finish-r11.blend` SHA `da320df0186316afa852be1f8748ba2054277628f28e18f00bd2fd02c95cb6c5` 已保存，腕部三姿通过、每手45个甲面内部点调整，近景未看。真实整合包 `refined-hand-package-r11.json/.npz` 已捕获，JSON SHA `d88230214a0295882bcc2742a6ec4ce79efd1a06886b7aff8631621cc2369a16`；尚未合入当前头脸源。

本次MCP重新连接成功，确认交互Blender仍打开旧9fc lower-face且dirty=true。仅暂停了耗算力的旧动画播放，保持文件、frame1和未保存修改，不保存/重载/关闭该会话。主源仍3d0e397a；正式导出、完整新绑定、口腔/视线、全程表演与新毛发物理均未完成。

## 历史记录：R11 原画发型继续修形：v8已保存，眼角和最终外观未通过

最新独立续作源 `work/reconstruction-r02/reference-adult-front-groom-v8-r11.blend`，SHA `cf8fbec30e7cabb58ed31c2bd2cf0ba74abe0e3effcc162b1d3b94291cc6d59a`，接续下方662054温柔笑/领片/下颌源。实际26046根评估毛发检查通过，原生显式选择裁短仅作用于目标冠前刘海层，未选曲线及全部根部保持。原画黑色低双马尾、原绑位和尾束继续保留；不以生成参考替换设计。

整合者已实际查看 `adult-full-groom-v6-preview-r11/full-groom-front.png`：额顶裸露带和孤立黑刷改善，中央主刘海仍过厚、主要露额缺口在画面左而原画在右、左鬓上翘，尚未外观采用。v8只修真实碰撞/未选曲线保护，不能据此视为已修上述造型。v6预览源1ae834仍禁止用于正式整合。眼角v3在完整闭眼的三角化比较处停止且未保存；v4将记录真实面/角点/边连接与三角化差异，尚待实际结果，不能提前宣称完全闭合。

独立手部源 `reference-hand-wrist-clearance-r11-v2.blend` SHA `6945a25f17edf3e3c081a67445648de52f1c22da63d3449e495a66e25c9e00a2` 已实际保存，235个旧前臂局部点收进新手；中性、屈腕25度、伸腕20度实际包含/间距检查通过，真实面连接不变。保留内部闭合旧帽，未声称腕口焊接，近景和新脸源整合尚未完成。足部真实观察 `foot-asset-observation-r11.json/.npz` 已执行，足部适配尚未安装。

灯光A/B首版把9个常驻修形误判为表情而停止，没有生成对照图；正在修准确的中性状态识别。后续绑定固定采用最终静态源新观察/新计划，然后 `binder → gaze → jaw → oral`，避免旧cranialParent报告误挂上牙。主源仍3d0e397a，正式导出未更新，外观及交付未完成。

## 历史记录：R11 当前源662054a7：领片、下颌和新温柔笑已保存，正在看图

当前独立续作源为 `work/reconstruction-r02/reference-adult-warm-smile-r11.blend`，SHA `662054a7e58ea65ae575b99ad8b2c3257c6dfdd013296a1bcfc2f6cd63590e18`。接续43eef27，包含内虹膜完整矩阵版055a4e6（外球原始/世界坐标不变）、领片v2 151641fa、下颌v4 7defec78及新的 `R11 warm closed smile`（默认0）。下颌实际两眼/两耳/口腔/颈接口位移0，侧上颈0.252mm连续收形明确记录，并非隐藏为零；原圆下巴仍保留。当前完整头颈蒙皮/新gaze仍未安装，必须在静态几何确定后重做观察与计划，旧491b静态计划不能直接证明新下颌。

055a源的 `current-face-smile-r11/` 两张真实图（0/旧natural笑1）均已看：旧笑太弱，未达到原图温柔神态。新warm笑口角目标3.85mm、颊2mm，`warm-smile-front-r11/` 的0/.5/1三图已真实完成并查看；满值嘴角上提可见，可继续制作，但眼睑仍僵厚，未达到整体外观目标。领片v2正面原先大片硬色块已消除，仍待侧面/动作审阅。`blink-closure-corrected-r11/` 实图显示中央闭合改善，两眼角仍有黑色裂隙；只读 `adult-blink-residual-r11.json/.npz` 已实际完成，下一修正按实际L3球面而非理想球。

最新闭角安装入口为 `adult_blink_corner_install_r11_v3.py`（1774aab6…），不是先前v2。v2错误比较整个旧观察箱导致失败；实际 `adult-blink-corner-dependency-diff-r11.json/.npz` 证明新源仅低处240点发生最多6.03微米变化，而非零字段、incident面、两环、2mm空间halo共13574点全部严格无变化，双球及头矩阵也完全一致。v3先保存实际差异再按真实依赖验证，0.2微米门槛未放宽，等待实际安装。v7刘海裁短失败：作者旧Trim组改动了全部未选曲线，并造成38点入肤；v8正改为显式选择的原生裁剪。显式v6诊断预览1ae834f5保留已知11条入眼问题，禁止当生产源。

毛发v6尚未保存：实际26k密集发丝中11条进入眼包络，最深约2.50mm；原画连续前发/错落分束布局正在修这部分，不改低双马尾设计。显式preview若生成，禁止视为生产源。`reference-adult-rounded-jaw-preview-r11.blend`（1767e549）仅是旧v3诊断，不要用它覆盖已通过的7defec78。当前主源/正式输出仍未更新；实际保存链在project-state的 `savedStageLedger`。

## 历史记录：R11 当前源43eef27：原画双马尾约束重申，发型v5仍拒绝

最新已保存独立源为 `work/reconstruction-r02/reference-adult-neck-surface-r11.blend`，SHA `43eef27aa61adaf8af5ba00faf73f9bdcc30d8ea5409d891ce5769dca4ce8ef6`。它包含v5毛流66213681、统一头尾材质b8b858b5、闭角修正a4dffd57及桥材质连续化。闭角露球从约28mm²降至左右5.188/5.312mm²，仍未完全闭合；源中默认blink/自然笑均0。颈材质已保存且原头/身体/桥几何保留，尚待新图确认。

`adult-full-groom-v5-r11/full-groom-front.png` 是b8b858实际完整发型图，已看并拒绝：前冠露皮、齐切厚刘海、侧面毛帘及尾弯仍不符合原图。修技术错误并不授权更换发型。用户再次明确必须保留原画黑色低双马尾、耳后扎束、刘海及脸侧碎发；面部按具体角色的温柔成年日本/亚洲女性方向，不能套用族裔公式。此约束已写入 `project-state.json` 的 `artDirectionLatest`，原动漫是最高身份参考，生成三视仅辅助。

下颌2mm收形首版因保护域0.252mm伴随位移停止，未保存；眼内虹膜首版因非统一缩放假设停止，未改网格。两者正在独立诊断/修正。详情与源/图片哈希在 `groom-closure-progress-r11.json`。正式主源仍3d0e397a，交付和外观均未宣称通过；以下均是保留的历史阶段。

## 历史记录：R11 实际表情候选 b0c708：眨眼已安装，但闭合外观拒绝

最新已保存续作源为 `work/reconstruction-r02/reference-adult-natural-smile-r11.blend`，SHA `b0c7089ca09022bd59ae6b1068b68f78f46629ad87722f60ef54df5aa0858447`，接续491b桥源并安装双/单眼眨眼和自然微笑，默认均为0。前一眨眼源为28250e7948af…。实际GN单精度输入与逐步单精度参考消除了旧双精度参考的误报，原2微米门槛未放宽；这不证明眼睑闭合。

`previews/reconstruction-r02/blink-closure-diagnostic-r11/frame-0001.png` 已真实渲染并查看：中央接触，但两端明显露球，闭眼线过深U形，上睑仍厚，因此闭眼外观拒绝。对应源b0c708与图片哈希记录在同目录JSON。自然笑实测能带动脸颊/下睑，但局部唇交点仍需有符号深度核查；自然笑近景尚未完成。下一步使用独立 `adult_blink_closure_r11.py` 修眼角，不能把旧表情结果当作成功。

毛发v4实际分区插值将头皮最大修正39.17mm降到9.75mm，最终密集点未穿入头皮；但两层作者聚束仍跨组488/145条并偏移发根8.97mm，失败未保存。v5正在串行执行，明确旁路错误分缝/聚束而保留旧节点。领口首次修正因左领片翻折拒绝；腕修首次因姿态三角化变化停止，独立v2严格比较原多边形连接，未通过前不可称修好。具体实存证据见 `performance-progress-r11.json`；主源、正式输出和外观验收均未变更。

## 历史记录：R11 当前整合源491b：脸颈修改已保存，发型v2外观拒绝并继续修

当前独立制作源为 `work/reconstruction-r02/reference-adult-upper-neck-seam-r11.blend`，SHA `491b6c85c9c5261e619bba14a5b622bd09dc237d52eb2474d26509f0eb5ba707`。它接续9fc缩短下脸，加入窄颈695e2b、上唇9d391c、柔化鼻fd4f42、黑棕眉d4e5face、donor重梳v2 d6bbda、C弯尾c55bcc和新上颈桥。新上颈桥2592点/2720面，底部224点保留；首次因内建sharp/custom_normal属性停止，v2明确保留/重算后实际保存成功。它还没有新的真实渲染，底口接线与头颈真正蒙皮未完成。

**c55的正/侧/quarter实际图片均已查看，发型v2明确拒绝：额前露皮、刘海卷成立刷/交叉团、发圈前放射毛刺、后脑鼓包。** 不得把guide通过当作最终发型通过。`donor-gn-stages-r11.json/.npz` 已实际导出18级GN，Roll与Big Clump有数厘米级推移，原始后脑guide也存在过大体积；正在改后处理和原始后脑体积。491b保留这套失败发型供继续修，不是外观已采用。眉鼻唇独立中间源 `reference-adult-brow-finish-r11.blend`（d4e5face…）仍保留。

第一次眨眼安装因鼻部塑形使旧广义眼ROI变化而正确拒绝，未保存眨眼源；当前正从491b重新导出当前形状观察再安装。旧c2观察保留，不能放宽兼容守卫冒充新数据。新尾形已撤销旧原生尾发回放，身体/衣服旧动作仍保留，但不证明新头发/头颈的运动通过。交互Blender另一个预览流程已打开9fc，当前聊天未覆盖该会话。详见 `static-assembly-progress-r11.json`；主源仍3d0e397a，正式输出与外观验收状态不变。

## 历史记录：R11 下半脸已实际完成首轮修形与三视检查

最新已验证保存并看过三视的独立源：`work/reconstruction-r02/reference-adult-lower-face-r11.blend`，SHA `9fc66ef37f9fbda70cb5260dfc81344b43da7ff6888e7a31372c425cf484a379`。`adult-proportions-r11/` 的正面、quarter、侧面真实图片全部完成且源/图片哈希对应。实际唇部上移0.800mm、前下巴上移8.992mm；眼/耳无位移，颈边界误差0，唇接触最大增量0.399微米。此前保护域失败已区分真实鼻体与应允许连续变形的下颊；鼻体边缘插值位移2.892微米明确记录，鼻部容差10微米，眼耳上脸/颈保护保持2微米，未改形状参数。

三视实际取舍：缩短下脸可保留继续制作，鼻梁/鼻尖仍直而前突，上唇与下垂口角不够柔和，眉睫可读性弱、眼神仍僵。窄颈/上颈桥/上唇/发型v2另行执行中，不以本段证明其成功。首版donor restyle失败未保存（guide2旋转路径需推离124mm），v2改为真实头皮曲面梳理并分开刘海与侧帘长度。原版与失败日志保留。

新增 `references/generated/realistic-character-expression-board-r11.png/.json`：已生成、查看并保存完整提示词与来源，四格为中性、闭口笑、笑着闭眼、轻微露齿。是造型参考，不是实际模型或物理证明；内置生成工具未返回实际型号，不能称已确认image2.5。详见 `lower-face-progress-r11.json`。主源/正式输出/客户验收状态仍不变。

## 历史记录：R11 当前接续：眼睛比例与真实 donor 毛发已保存

当前独立续作源为 `work/reconstruction-r02/reference-adult-eye-proportion-r11.blend`，SHA `c2afb5d4499ff44475b57f09853c1d64d2d22b39bdab71b97f42e801ef605553`。它包含实际适配的 Daniel Bystedt 原生引导/GN 毛发（e10888af…）、约24.6mm眼球和相应眼睑比例。原眼中心保持。实际 donor 正面与 quarter 已查看：毛流层次可用，但仍为散乱短发，未完成原画刘海/双马尾；发圈浮空、鬓角暗块仍待修。新眼比例的实际画面尚待生成，不能用旧 e108 图片证明新眼外观。

本次重新核对发现先前下半脸执行在保护面位移检查处停止，未生成 shortened-face 源或 `adult-proportions-r11` 图片。当前重试诊断已明确：双眼和耳实际位移为0，低颈仅0.334微米；超界点位于被宽泛区域框包含的下侧面颊，需按真实解剖区域修正检查并审视连续形体，不能直接放宽眼耳颈保护。两个新只读观测已实际完成：`adult-blink-current-size-r11.json/.npz`（数组SHA `8328fb47…`）与 `adult-neck-precision-r11.json/.npz`（`2b53c286…`），均对应c2源。眨眼观察CLI只有 `--source/--output`，没有 `observe` 子命令。

128样本单眼睫毛光学诊断已完成：粗糙度0.5下实际可见黑色细丝，0.85没有明显可见收益；保留0.5。此前双眼图的发白不能笼统归因降噪或宣称已解决，丝束排列和远一些的可读性仍需修。详见 `eye-donor-progress-r11.json`。本轮已实际启动10个子代理并分开脚本所有权；大型 Blender 修改/渲染仍由整合者串行调度。MCP本次连接不可用且无交互Blender进程；后台Windows Blender实际加载/计算成功，没有覆盖任何用户打开的场景。

主源仍为R09 `3d0e397a…`，正式导出未更新；新表情、手部整合、衣服与毛发动态、连续视频、实时交付仍未完成。以下进度为此前阶段记录，不得据其旧候选或“正在执行”文字覆盖本段已核对状态。

## 历史记录：R11 成年女性头脸重构（2026-10-28，最新）

**最新继续制作源为 `work/reconstruction-r02/reference-adult-low-neck-r11.blend`，SHA `3a6baaa6271917cdd0c4c2b75c374651262ddde62d7ec7cb88e849fb31bb9d7f`，仍未写主源。** 它接续504503深棕虹膜源，依次加入真实唇体淡玫瑰材质（8fa6d43…）、独立不透明睫毛膏候选（be47284…）、低颈收形与实际边界重建桥。包含此前真实女性头脸、自然眉、轻妆皮肤、错落睫毛v3、小笑key默认0、眼缘层、眉睫绑定、下颌与上睑静态key默认1。仍使用bb678d旧groom作上下文，不含被拒277676重梳布局。`adult-face-contour-r11/` 的504503正面/quarter/眼近景、`adult-makeup-r11/`的be472正面/眼近景与3a6口近景、`adult-low-neck-r11/`的3a6正面/侧面均已实际完成并查看。

低颈新源减轻肩台和后颈尖折，但颈整体偏宽、后方仍有接线/平面，未认定颈部外观通过。两次失败未保存，retry2确认Blender5.2给bridge换新mesh会清空组名表；按原6组名称/索引精确重建后才成功，原body/rig未修改。眼睛仍偏小、下半脸比例与唇形仍需改造；当前正在独立准备约24.6mm眼球和相应眼睑/眼角尺寸，并实测可见球宽与眼角间距，不能混为一个量。新眼尺寸成立前，旧尺寸blink和自然笑草稿不直接执行。

睫毛膏只更换4组真实睫毛材质，原数据与原始材质完整保留；绑定现在允许明确记录的材质合同，原位置/半径/完整属性/拓扑/实际GN输出仍严格检查。be472实际眼近景仍偏白，不能称改进。只读实际材质审计确认原始/评估均用新黑色Principled，红色Emission诊断确认可见丝就是当前对象；红图仅诊断，不是交付妆容。128samples同单眼的粗糙度/降噪对照正在执行。

眉睫绑定实际完成 `reference-adult-face-fibres-bound-r11.blend`（91c24768…），包括临时局部变形与恢复探针；原曲线/半径不变、零隐藏，中性最大误差约1.35微米。它提供跟随，并不证明眨眼、抬眉、转头和全部表情已经自然。颈桥 `reference-adult-female-neck-r11.blend`（98dd462…）消除大块额外肩片，但实际正/侧图暴露颈根两侧鼓包与后颈尖折/可见接线，正在对低颈和桥一起连续收形；不得把端点重合当外观完成。保持同一 L3 拓扑时新眉睫绑定可继续使用；若改变 Mask/拓扑必须重建眼缘图层和重绑。

`reference-adult-groom-refined-r11.blend`（27767661…）的三张实际图位于 `adult-groom-refined-r11/`，**重梳造型整体被拒**：粗亮冠柱、稀疏针尖刘海、裸露黑帽与直绳侧发。两个生成失败日志和成功但外观失败的源均保留。新路线开始真正适配 Daniel Bystedt 官方 Hair Styles 的作者导向、两岛 UV、插值/clump/noise/GN；实际原资产孤立图 `hair-donor-native-r11/donor-quarter.png` 已完成并查看（11193评估曲线、211265点），原资产未保存修改，尚未映射角色。署名与网页原文 CC BY-SA（版本未明）须保留，不能当 CC0 或已适配成品。

用户新增授权：缺资源可以自主绘制，要求 image2.5。已使用 `imagegen` 技能和内置绘图工具；该接口不暴露模型型号，已明确告知无法核实2.5，元数据不能写成已确认。`references/generated/realistic-character-three-views-r11.png` 是根据原动漫生成的写实**设计参考**，不是实际模型图/扫描/验收；原动漫仍为身份主依据。`textures/generated-r11/iris-dark-brown-r11.png` 是实际1254×1254自绘虹膜候选（提示词要求2048，工具实际未给到），已经以sRGB内嵌到504503源，原R07图/材质保留。两份完整提示词、用途、模型未暴露和来源记录分别在同名JSON，不能把生成图作为照片级完成证据。

已完整读取用户更新的目标 `088fac4c-6196-462f-8f38-c88956ada2fc/goal-objective.md`，项目留存副本 `references/user-goal-objective-20261028-update.md`，SHA `9d6c347c2663c073cea766fb96ce13a8b25c63abe11920d0cac178a66b2a7c72`。继续自主完成原画角色的写实、自然表演与完整交付；用户对旧脸的明确拒绝以及精致自然日本女性/轻妆肤色的新增方向仍有效。主源仍是 R09 的 `work/character.blend`（3d0e397…），没有把新静态候选冒充整合完成。

新的实际静态头脸候选为 `work/reconstruction-r02/reference-adult-female-head-r11.blend`，SHA `080f5d5d6b214deace3148923028f7ffac1a12ce748d953008e2e2be79ec3e48`。采用同一官方 CC0 Human Base Meshes v1.4.1 中真正的 `GEO-body_female_realistic` 头颈，保留作者 Multires 3；新眼用该女性原始 24 mm 球体随整体约 0.9328581 缩放，复用自制棕虹膜，未沿用旧 32.65 mm 大球与压鼻变形。实际原始数据、L3和原头观察分别为 `female-basemesh-actual-r11.*`、`female-multires-actual-r11.*`、`face-structure-observation-r10.*`。新对象是 `R11 Female head study` 和 `R11 Adult sclera/iris L/R`，旧 `R01 Head` 只保留躯干显示；**旧的硬编码 R01 面部制作/导出脚本不能直接套用于新头。** 新表情、颈权重、口腔表演和实时适配尚未完成。

六张真实图在 `previews/reconstruction-r02/adult-female-head-r11/`：structure/groom 各 front、quarter、side。独立审阅 `adult-female-head-visible-review-r11.json` 明确 **neckBodyFit=FAILED、groomFit=FAILED、appearanceAccepted=false**。成人眼眶、鼻翼、人中和唇体明显恢复连续实体，这条基础路线保留继续制作；但新下颈/肩片在旧领口外露出、背后有锯齿切边，旧发帽被新额颅穿出且耳后悬空，绝不可宣称整体外观完成。结构图两个浮空发圈是隔离显示漏项。不能缩新颅骨迁就旧帽或裁图掩盖接缝。

较早阶段记录（已由上文继续推进）：眉源 `reference-adult-female-brows-r11.blend`（575c5ff…）有两张实际结构图；作者头颈Mask单一边界432点、旧体颈口224点。`face-contact-actual-r11.json` 当时实际测得眼球世界边界误差2.43e-8m、唇体中央最小间隙约0.127mm；160/320点眼唇拓扑环并非湿线/真实接触线，不可拿其约3mm距离强拉闭口。新组合仍需重新验证实际接触和动作，旧数值不作新外观验收。

实际贴头皮 groom 源为 `reference-adult-female-groom-r11.blend`（bb678dd…），保留所有旧 groom 数据。根层位于新颅骨外约 0.25 mm，尾发/旧缓存没改；这不证明新头碰撞兼容。其后轻妆源 `reference-adult-female-skin-r11.blend`（def05a42…）复制材质并新增色彩/粗糙度属性，原地图/光学/灯光不改。`adult-female-skin-r11/groom-front.png` 和 `groom-quarter.png` 已实际查看：旧头皮穿插明显减少，但**冠额造型仍被拒绝**，刘海过短、粗平发条和侧前方横向浮弯需重梳，不能按数值通过采用整套发型。脸仍需眼缘、神态和轮廓精修，颈根肩片仍未修复。

较早睫毛副本 `reference-adult-female-lashes-r11.blend`，SHA `9be0d305a1fa57a949df84f5b8db4087e8238d6feab05000118934239ddd7ab4`，在 def05a42 源上增加每眼 116 上睫毛、48 下睫毛。它实际落在内侧睑缘，近景几乎不可见，已由 v2 的真实外侧生长缘、再由 v3 的156上/48下错落分布修订；原对象/data 均保留隐藏。`reference-facial-fibres-r11.blend` 的 Huang 光学试验仍显浅色梳齿，不作为采用的光学改进；当前504503源沿用 v3 的原CHIANG材质，眼睫外观仍需改善。

首次 prepare 因骨父级依赖更新顺序失败；retry1 因把新增删除显示 GN 后的旧对象尺寸拿去与原显示栈比较失败。retry2 在临时关闭唯一新增 GN 的同条件下严格比较旧几何/keys/UV/material/rig，仅实际记录 `is_active` UI 布尔变化，成功保存。两个失败日志与所有历史源均保留，无新异机备份或全工程冻结声明。

## 历史记录：R10 当前制作与用户新增方向（2026-10-28，最新）

**优先级再次更新：用户已明确拒绝当前面部，要求精致、自然的日本女性面部。** 当前 R09 工作主源仅作为保留的制作基线，不能作为外观接受或交付依据。暂停旧脸肤色小修与最终视频/实时发布，先重新建立成年女性的眼眶、眼球、鼻唇、下颌和软组织关系。实际源代码显示早期 R02 曾鼻部后退 8 mm、局部眼眶纵向增加 90%、眼球放大 1.30 倍，R05 又整体放大头部与扩大眼裂；这些叠加是当前人偶感的重要嫌疑，须用实际观察与新候选验证。R01 原基础同样偏男性、长脸，不可简单回退即宣称解决。最新完整记录是 `user-face-direction-r10.json` 和状态 `currentFaceDirection`。

用户要求十个并发子代理，系统实际同时最多三个，已明确告知并用满，分别负责整脸重塑、成熟女性人脸资产、独立外观审查。手部 v2 已通过真实蒙皮跟随检查并渲染双手背面，文件 `reference-asset-hands-r10-v2.blend` SHA `7f45d5a151c8b5efb61afda46d482d97dc60ed5774a988847d82cd6447dd9b4b`；未做完整屈指/掌心检查，未合入。旧 9af 手候选实际缺权重、停在 T pose，保留失败历史。马尾新静态 helper 仅离线计算，尚未执行；皮肤只读观察已完成，旧脸肤色草稿未应用。以下相同日期的计划如冲突，以本段面部优先级为准。

用户明确要求动作和细节符合物理规律，肤色为自然的亚洲女性肤色、轻薄底妆后可以略白皙，并要求尽量使用子代理并发修改。继续自主制作，无需逐项确认。三个子代理分别推进手部/屈指检查、毛发资产/毛流、肤色/妆感；整合者独占主源写入，并在当前有限硬件上串行执行重渲染。

当前固定主源仍是下述 R09（SHA `3d0e397a6ba0b8742054718d01c4eed5b4ff0daa4a603fac43aa020b87707166`）。已实际补拍 `master-details-r09` 的双手、眼口和侧面，以及 `shoulder-portrait-r09/shoulder-portrait.png`。近景显示手背平整、指节不足；整幅头肩图显示马尾 S 弯过硬、末端偏长尖；面部仍有明显 CG 感。技术校验不能替代这些实际缺陷的修复。

`reference-crown-volume-r10.blend`（SHA `3fe0565d295e74593f61f8d9eb727db2d4664dd4369652690ab68037a3491cce`）已拍三视角并**整体拒绝采用**：扩大不透明发根层形成帽檐状突起，刘海回折。保留原 PN 贴头皮根层，改由长发路径与自由中段形成体积。决定记录在 `crown-visible-decision-r10.json`；不得用这个被拒候选作为后续主源或实时适配基底。

现成 CC0 Human Base Meshes 的真实手部网格已实际适配到 `reference-asset-hands-r10.blend`（SHA `9af94e0ea48fae0bbe98780bda17376b8ba4f483f48fce8effb58e9fa704231a`），保留原骨架和各手原有组索引。首次准备因左右组顺序假设错误而停止，已修复按骨名匹配；第一次近景相机法线校验失败，正在诊断，尚无外观通过或整合声明。不得称该资产为扫描手，现有来源没有证明采集方式。

官方 Daniel Bystedt `Hair Styles` 示例已下载到 `assets/blender-hair-styles-bystedt-2023/`（SHA `1ad6202095c1793678fee7d69a7e9f8b5fdb6e5c293d300eb1062d2d437e8d48`），已在禁用自动脚本的独立进程读取清单，尚未导入角色。网页及内嵌说明均写 CC BY-SA，未标具体版本；不能擅填版本或删去署名/相同方式共享条款。真实结构是 346 条主引导发、30 条额外长发与 GN 插值，用于评估更自然的冠部层次。最终源稳定后才重新生成连续视频与实时适配，旧缓存和报告不得自动覆盖新发型、手部或材质。

## 历史记录：R09 工作主源已更新（2026-10-28，最新）

`work/character.blend` 已由唯一整合者更新为 R09，SHA-256 为 `3d0e397a6ba0b8742054718d01c4eed5b4ff0daa4a603fac43aa020b87707166`。它来自 `work/reconstruction-r02/reference-production-refined-r09.blend`（SHA `1c0b0210114170f0ce6e09b266b4a9ded58f29bf1df45a6e2be13e9b5c77aa72`），保留匹配的 301 帧身体动作、原生头发和裙摆。暂存源与安装后的固定主源均已分别在新进程重开，通过头部、UV、材质、动作、依赖和有限帧回放核对。实际证据在 `project-state.json` 的 `masterIntegrationR09` 及 `work/master-integration-r09/r09-refined-v2/master-reopen.json`；这不等于连续运动、实时版或外观完成。

旧迁移主源及项目状态已复制并校验保存在 `work/master-integration-r09/r09-refined-v2/`，较早失败尝试 `r09-refined/` 同样保留。第一次检查将仅含顶点、隐藏渲染的原生发丝坐标载体误判为缺材质表面；修正为严格核对该唯一载体的类型、零边零面、隐藏渲染和播放记录，其他可见表面检查未放宽。没有删除历史候选，也没有新的异机备份或全工程冻结声明。

本轮实际采用了轻微口周位移修正、牙冠接缝调整、局部下颌曲率，以及皮肤、眼白和虹膜光学改进。当前中性正面及 quarter 在 `previews/reconstruction-r02/appearance-refined-r09/`；同下颌/牙齿几何的 AA、EE、OU 真实图在 `lower-jaw-keyframes-r09/`。虹膜底层宽反光去除后棕色更清楚。椭圆刘海试验因根部暗槽和宽发带感被拒绝，最新源已经恢复较早 small-flow 刘海，并保留后脑 PN 曲面改进。取舍记录为 `appearance-visible-decision-r09.json`；这些是代理决定保留的工作改进，仍有明显 CG 感，未达到照片级/AAA/最终外观验收。

下一步补拍当前主源双手、眼口、正侧面与灯光细节，运行新 `action_review_sources_r09.py` 的表演/步行独立副本并生成连续视频；`final_detail_review_r09.py` 保留真实第 1 帧原生缓存，不使用旧静态裙或旧手图替代。旧手图来源仍为迁移主源，不可证明 R09 手部通过。实时源仍需从最终采用源重新准备、烘焙、导出和实际查看；固定 `exports/avatar.*` 仍是历史文件，当前 `.charakit` 状态未覆盖模型。下方“主源未更新”等段落均为较早历史状态。

## 历史记录：R09 当前整合与实际外观取舍（2026-10-28，更新）

当前可继续制作的组合源为 `work/reconstruction-r02/reference-body-combined-r09.blend`，SHA-256 `6f7d02e9b8dd5f712987726d6798f994c7713ae86a602754b6bad721095839ef`。它在 R09 缩短下半脸、补齐后脑毛发和后脑根壳曲率修正之上，整合了 301 帧新身体动作、匹配原生马尾回放和新布料。重新打开的检查为同名 `.blend.reopen.json`；五张真实姿势图在 `previews/reconstruction-r02/body-combined-sparse-r09/`，下蹲、弯腰和抬臂已查看，之前的大块裙摆穿腿未再出现。连续视频尚未渲染，主源和正式导出仍未替换。

后脑曲面修正采用 `reference-root-flow-r09.blend`，SHA-256 `b7ebc1ede9b9e73483b6835945630280bf4765c70df620317bf9c3da8b0ca0b0`。原壳继承低细分头部平面，同步运输 root/scalp/gather 后，实看背面大平面带明显减少。`reference-forelock-volume-r09.blend` 的较大刘海改形**未采用**：前/侧前图出现宽平发板与平冠高光。保留历史，不要按文件时间选它作为新主源。取舍与对应图见 `groom-visible-decision-r09.json`；其中后脑图仅用于后脑取舍，前刘海仍回到较早小修。

R09 新表情关键帧仍暴露 EE 的方下巴和 OU 的尖楔状口周；`perioral_delta_r09.py` 正在独立候选修正唇外皮肤位移场。首次执行在只读修改器属性快照遇到不支持 IDProperties 的类型，未写源，日志 `perioral-flow-r09.log`；不得把离线数值改进等同实际外观通过。眼色和牙齿适配候选尚待实际制作与看图。

实时版 66 mm 袖部包络与逐段发束宽度的参数试验，在实际本地查看器明显减少肩部截断；较小胶囊方案反而抬发更多，已拒绝。采用试验文件 SHA 为 `baede9f103b4dc5ac2cb1c27eab40505526152667eabe40a4cc6bff4c635c28d`，仍有下段尾发/胸衣擦碰及前抬。此参数尚未写入最终编辑源；`runtime_integrate_driver_r09.py` 已准备，但须待最终制作源确定后再实际更新 face/scalp/fringe/gather/root、重烘 skin/root/iris 并导出与查看。旧根壳 normal 不得复用。

后台 Blender 仍由整合者串行执行。临时浏览器预览和其服务器已关闭；交互 Blender 为空场景，仅用于有限读取，没有替换主源。没有新增异机备份、AAA/照片级或客户验收完成声明。

## 历史记录：R08 实际查看器检查后继续 R09（2026-10-28）

以下更新优先于下方历史进度。R08 的 80 帧连续步行已完成并编码为 `previews/reconstruction-r02/combined-walk-continuous-r08/walk-preview-r08.mp4`，对应 `reference-face-flow-r08.blend` 的已记录 SHA。181 帧表情组合源 `reference-face-performance-r08.blend` 已保存，11 张真实关键帧已生成；其连续视频尚未生成。

当前制作续作副本为 `work/reconstruction-r02/reference-production-uv-r08.blend`，SHA-256 `1e94ff32a0c0389e78c67642ebdec0f8887a23933f61b3134eb64795a55420ad`。它在 R08 上仅新增独立生产 UV，原几何、形态键、权重、材质和旧 UV 状态不变，证据为 `production-uv-r08.json`。主源与旧正式导出尚未替换。

R08 运行版已成功完成材质烘焙、面部替换、实际 VRM/GLB 导出及本地浏览器加载。源为 `work/runtime-r06/r08-export-ready.blend`；预览 VRM 为 `work/runtime-r06/preview-export-r08/review-preview.vrm`，SHA-256 `184e697e1d8e86512641f6e9da1baed154eed6e3a2151b912d3075d12b8ef2b5`。实际包含 169 个网格节点、832496 个三角面；查看器将三个头部材质拆为 171 个渲染网格。17 个预设及4个自定义表情存在，实际检查了微笑叠加闭眼、张口、转眼和噘嘴。

**该运行版不可作为已通过的成品。** 实际侧面摆动时马尾进入肩部；已定位两条新尾链遗漏关联现有肩/背碰撞组。背面呈明显分段黑帽；制作源近景 `source-headback-r08/back.png` 也确认后脑中央漏发。下半脸相对原画仍偏长。具体证据与截图见 `work/runtime-r06/runtime-viewer-observation-r08.json` 和 `previews/reconstruction-r02/runtime-viewer-r08/`。R09 正针对这三处问题制作可逆修正；不能用技术导出成功代替这些可见问题的修复。

R09 更大幅度身体动作正在独立副本验证，尚无匹配新布料/头发缓存。旧 R08 fullbody 重新打开的原生回放数值检查已完成；旧/新碰撞体比较因旧 R04 下颌驱动无效停止，不能宣称碰撞等效。不要覆盖历史缓存，也不要把旧小幅动作结果套到新动作。

交互 Blender 仍为空场景以节省内存，后台 Blender 只由整合者串行运行。没有新增异机备份、AAA/照片级质量或客户验收完成声明。

## 历史记录：R08 当前续作入口（2026-10-28）

已采用 `reference-face-flow-r08.blend`：在 R07 已匹配的步行组合上局部缓和脸颊到口周的硬折线，并调整现有眉丝排布和渐细；原眼睑接触点、唇缘、颈部、骨架和原始头发不变。两个实际近景在 `previews/reconstruction-r02/face-flow-r08/`。来源 SHA 及当前候选仍以 `project-state.json` 为准。新的完整步行源已重新打开检查，`combined-walk-reopen-r08.json` 对应此 SHA；80 帧连续步行正在逐帧渲染，未完成前不能宣称视频通过。

独立运行版最新已保存文件为 `work/runtime-r06/r07-material-sock.blend`，包含 R07 运行几何、原生发丝投影发片、原始脸颈/口腔保护的衣内身体裁减，以及已实际烘焙的发根、上衣、裙子和袜子贴图。R08 面部替换入口 `runtime_face_refresh_r08.py` 已准备，尚未执行；需要保留这些已烘部分，只重新采样 16 个面部对象并验证骨架一致，再烘新皮肤。最新裁减已修复并验证 `sharp_face` 等属性，首次失败记录保留。后续仍需领口/腕口动作画面、真实导出和实际查看器检查。

本机内存曾接近用满。交互 Blender 的旧 R07 预览经确认无未保存改动后已释放为空场景；源文件没有改变。当前后台渲染须结束后再启动其他大内存 Blender 制作。恢复可视预览时明确打开上述 R08 或后续经过检查的候选。主源、正式导出和 `.charakit` 尚未更新；没有新增异机备份声明。

## 历史记录：当前推进至 R07：动作组合与交付适配（2026-10-28）

以角色状态 `reconstructionR02.currentSavedCandidate` 的明确路径和 SHA 为续作入口；以下较早阶段记录均保留作历史。主源 `work/character.blend` 和旧正式导出尚未替换。用户已授权自主决定方向和常规制作，不再等待逐项确认。

R06 已补齐头皮毛流、低双马尾根部连接与发际线透明过渡。R07 已修复实际眉毛形态键组合埋入皮肤的问题，并新增局部颊面连续性修正；可见结果在 `previews/reconstruction-r02/face-continuity-r07/`。另一个全局减弱脸部塑形的方案在 `face-volume-study-r07/decision.json` 明确未采用。虹膜已改用工作区独立程序生成图 `textures/procedural-r07/iris-brown-r07-v3.png`；旧来源不明纹理只作为历史保留，不应带入新的发布材质。

头颈 REST 支点已校准：原颈段仅约 3.6 mm，现约 111.1 mm；原始几何与附件静态位置保持在微米级误差内。实际抬低头图随后暴露颈部蒙皮过早跟随头骨及后颈体积外鼓，已分别用 `neck_weights_r07.py`、`posterior_neck_contour_r07.py` 修正；最新侧视结果位于 `neck-contour-poses-r07/`。全局减弱脸部塑形的试验未采用；脸颊口周和眉丝仍在继续审阅。

CMU 07_01 步行 v4 已完成 631 个实际小数帧足底检查；前视头姿为单独派生动作。步行专用原生布料已计算，139 份缓存保留；外部缓存读取与实际存储坐标不一致，恢复尝试失败。现已将缓存中的真实 80 帧顶点内嵌为原生形态键回放，并在 `reference-combined-walk-r07.blend` 中合并匹配双马尾动态和手工眨眼。重新打开后的裙摆最大坐标误差约 0.358 微米，头发采样整数/小数帧本地坐标误差为零；证据为 `combined-walk-reopen-r07.json`。`combined-walk-sparse-r07/` 四张真实姿势图已检查，连续动作仍待完成。原模拟设置、失败缓存和历史数据均保留。

运行版几何适配原型在 `work/runtime-r06/`，不是正式交付。原始共享解析透明图的发片近景明显过稀，已判定外观不合格；新五区域原发丝路径覆盖投影已实际生成并检查，密度和发型轮廓明显改善。最新版脸颈和虹膜已生成 `r07-source/face/hair/geometry.blend` 独立副本。衣内身体裁减候选首次因 `sharp_face` 属性丢失而停止，未保存；必须修复并重查。材质烘焙正在实际执行，PBR VRM/GLB 导出与查看器验证尚未完成。制作源保留完整身体、曲线毛发、物理材质与原生模拟。

当前仍是执行中的制作目标；没有照片级、AAA、客户验收或新增异机备份完成声明。

## 历史记录：当前制作：按原画重构 R02 至 R05（2026-10-28）

用户提供原创动漫头像后，要求尽可能还原角色，并明确授权由代理自行决定外观与技术细节、继续推进交付，不再等待 A/B 方向确认。原画保存在角色目录 `references/user-original-portrait-2026-10-28.png`。下文 R01 的“等待方向选择”只代表历史状态。

当前新增制作仍在 `work/reconstruction-r02/` 独立副本进行，确切续作文件及 SHA-256 以 `project-state.json` 的 `reconstructionR02.currentSavedCandidate` 为准。主源 `work/character.blend` 和已有导出暂未替换。不要按文件时间自动选择来源，也不要把 R01/R02 的过程脚本顺序重跑：部分眼睑接触和虹膜调整已保存在副本中，早期脚本不是一键重建入口。

已完成颈部与身体的拓扑连接，并修正连接过渡误碰手臂权重造成的袖口穿插。R04 对应副本 `reference-native-hair-r04.blend` 已将 301 帧手工全身检查动作、匹配裙摆外部缓存、双马尾内嵌形态键与 Geometry Nodes 组合保存；重开后发丝整数帧与小数帧逐点对照误差为零，四张组合姿势图位于 `previews/reconstruction-r02/combined-r04/`。这里只覆盖该段手工动作，不能自动套用到其他动作。

当前 R05 正在继续原画相似度调整：头部及眼球、口腔、发型附件统一放大 1.10 并降低 15 mm，颈部过渡严格限制在中轴；增加更圆的睁眼轮廓，并限制眨眼与眯眼叠加。继续来源必须读取上述状态文件的明确路径。`reference-eye-aperture-r05.blend` 是已发现新增形态键基底错误的过程副本，不作为续作入口；修正副本名称带 `eye-aperture-fixed-r05`。R05 修改几何前已经撤下 R04 双马尾缓存并保留其历史引导数据；发型调整结束后需要重新烘焙，R04 组合报告不能证明 R05 动态通过。

四段原有动作及 R03 表情/全身检查动作是手工编写的动作。另已取得官方 CMU 07_01 真人步行及来源许可记录，并生成独立重定向预览；首次鞋底检查发现摆动脚跟擦地及接触突变，v2 正在修正，尚不能称自然步行通过。R03 连续脸部视频属于旧源；R04 全身连续视频正在渲染。当前尚无新版最终交付视频。

真实渲染位于 `previews/reconstruction-r02/`，过程图可能包含已知缺陷。当前尚未达到用户目标中的照片级最终交付质量，用户并未作外观验收；新版毛发与表情也尚未完成实时 VRM 适配。当前目标仍在执行，不能把已保存、已绑定或已烘焙写成完成交付。没有新增异机备份完成声明。

## 历史记录：2026-10-28 写实重构首轮（尚未合入）

`work/character.blend` 仍是原迁移主源，本轮未覆盖。新的静态方向副本为角色目录下 `work/reconstruction-r01/candidate-review-A.blend` 和 `candidate-review-B.blend`；A 为较明确的下颌，B 为较短、柔和的下半脸。当前 Windows Blender 打开的是 A 副本。

本轮采用官方 CC0 Human Base Meshes v1.4.1 动画头与眼球作适配基础，重新梳理曲线双马尾，并以可逆形态键收拢裙摆。对比在 `previews/reconstruction-r01/`：`before/`、`review-A/`、`review-B/` 是同机位同光照的真实渲染，其他子目录为迭代记录；仅带 `comparison.png` 后缀的拼图和 review-A/B 为本轮审阅入口。来源、哈希、路线与限制见 `work/reconstruction-r01/asset-research.json`、`review-summary.json`、`static-checks.json`。

这些副本是静态外观提案，尚未绑定新头脸/毛发，未完成颈肩焊接、皮肤精修、表情或物理。新脸的成熟度和气质需要用户判断，不能称已保留年龄特征或已达到 AAA/照片级。旧 VRM、GLB、动作与报告继续对应原主源，不能用于证明新副本通过。新候选未合入，也未建立新的异机备份。继续前先读取本轮报告和用户的方向选择，勿自动覆盖主源、重新运行早期生成脚本或把这些静态副本作为动画成品。

6 个 `reconstruction_*r01.py` 为本轮新代码，保留在工作区制作源码目录中；官方资产保存在 `assets/human-base-meshes-v1.4.1/`。恢复制作需先在候选中检查，固定工具包没有改动。

Windows 续接更新（2026-10-28）：原生 Blender、VRM 插件和 MCP 已安装，并通过正确主源的实际 MCP 读取验证；主源哈希未变。具体安装路径、验证范围与桌面工具重载说明见 [Windows 安装验收记录](WINDOWS_SETUP_RESULT.md)。下文保留迁移交接时的原记录。

本文件适用于新的 Blender 工作区，使用 `main` 分支。原制作会话于北京时间 2026-10-28 18:15 完成交付；完整换机包的文件清单生成于 18:03:44。

## 历史记录：接收结论

原交接包与本工作区的主 Blend、当前 VRM、`project.charakit` 是同一版本。接收时重新计算了这三个本地文件的 SHA-256，均与原交接清单相同。

原包中与此前迁移清单共有的 962 个角色资产，其清单大小和 SHA-256 全部相同。唯一新增的角色资产是 `work/qa-comprehensive/migration-dependency-audit.json`，现已补入。没有重新建模或替换主模型。

原会话的恢复报告记录：完整包在本机另一个路径解压，1,836 个文件校验通过，应用恢复成功，Blender 打开主工程成功，19 张文件图片全部内嵌，128 根骨骼和零外部链接库。这是原会话完成的本机换路径测试，新电脑仍需实际验证。

## 历史记录：主文件与当前导出

以下路径相对于 `projects/6c8500bf-1c38-481a-babd-e5478a50dd67/`：

| 用途 | 路径 |
| --- | --- |
| 唯一建模主源 | `work/character.blend` |
| 整理后的可编辑导出 | `exports/avatar.blend` |
| 当前 VRM | `exports/expression-review/avatar-preview.vrm` |
| 当前 PBR GLB | `exports/expression-review/avatar-preview.glb` |
| 与 VRM 对应的运行版源 | `exports/expression-review/avatar-runtime.blend` |
| 当前验证总表 | `exports/expression-review/avatar-runtime-review-summary.json` |
| 当前视觉问题 | `exports/quality-assessment.json` |
| 本轮 23 张实际渲染 | `previews/refined-current/` |

主源 SHA-256：`6e7361e7f5c306325d3f33c4c9f6f56b1a489d5f4579bb4be951d4c3d59cc8c1`。
当前 VRM SHA-256：`09c82321c484e5512607c9218b442bf998d5076be0d48db574970668e8617c99`。
完整资产身份及其对应关系见角色 `project-state.json` 和根目录 `asset-manifests/`。

## 历史记录：已完成操作

身体与衣物间隙、耳朝向、前额细丝、后脑和后颈头发的本轮修改已经整合。最终交接明确：当前没有待合并的必需修复；此前未合并和未保存场景的疑虑已经解决。历史工作副本仍完整保留。

`work/local-model/finish_comprehensive_refinement.py` 已经在旧主源上执行过一次。不要对当前主源重复执行它，也不要盲目重跑早期整体生成或头型变形脚本。它们是制作过程记录，不能把所有脚本依次执行当作恢复步骤。

`.charakit` 已登记当前表情预览，原制作流程阶段有意保留。应用显示较早阶段时不要重新生成 Tripo 候选。

## 历史记录：旧交接包与新工作区的关系

`Charakuru-workspace-2026-10-28.zip` 是约 6.24 GB 的原布局完整备份，保留用于恢复后备。其记录的 SHA-256 为：

`10f0a778d7109459427b45d65d87e93bd0cde59e231c71119c1015f7ea7fa0a3`

本次没有重新解压或逐字节重验整个 6.24 GB 包；包级验证沿用原会话的交付和恢复报告。本次核对清单，重算三个主文件哈希，并对提取的新增报告及安装文件分别校验。

该包包含旧 `.git`、根目录代码、配置和会话状态，不能整体解压覆盖 Blender 工作区。原包中的启动命令和续作提示词针对旧 Charakuru 布局。新机应使用本仓库的 [迁移步骤](MIGRATION.md) 和 [续作提示词](CONTINUE_PROMPT.txt)。

原包的运行代码与固定工具版一致，但有部分工具文档与本仓库锁定发行版不同。继续使用 `tools/charakuru/1.1.0` 或通过锁文件安装，不用旧包根目录替换它。新增依赖审计中的旧路径和旧布局建议仅作历史证据。

## 历史记录：新机环境

完整软件清单、安装顺序、平台路径示例、Codex MCP 配置与验收见 [新电脑环境安装与验收](NEW_MACHINE_SETUP.md)。手动搬运的准确范围、源路径和目标位置见 [迁移步骤](MIGRATION.md)。

恢复基线为 Blender 5.2.3 LTS、VRM Add-on 4.10.2、mcp-for-blender 2.2.4，Node.js 要求 22 以上。工具安装和语法检查使用的系统 Python 最低为 3.8；MCP 2.2.4 自身要求 Python 3.10 以上。

`tools/setup/` 已保留 VRM Extension 安装 ZIP、MCP wheel 和 Blender 端 `addon.py`，大小和 SHA-256 记录在 `toolchain.lock.json` 的 `optionalInstallationMedia`。这些文件需要在新机实际安装；它们不包含全部 Python 依赖，不能视为完整离线安装环境。

Blender、Node 和 Python 按新电脑的操作系统及 CPU 架构安装。Blender MCP 连接保留本机回环地址，并按实际插件配置核对端口；不复制整份旧电脑的 Codex 配置和登录信息。连接后实际读取当前场景，不能仅凭安装成功认定可以制作。

## 历史记录：接续顺序

1. 按资产清单校验复制结果，确认新旧电脑使用同一 Git 提交。
2. 打开 `work/character.blend`，核对图片、骨架、表情、材质和物理；确认 MCP 读取的确实是该工程。
3. 启动本工作区预览，检查静态、眨眼、口型和短动作，再开始新一轮制作。
4. 先处理头脸全角度自然感、枕颈接合与发帽边缘，再改善头发、皮肤、手部和衣料。
5. 确定目标软件和设备后处理性能预算；当前运行版 614,282 个三角形，超过脚本 300,000 的目标。

模型仍为审阅阶段，用户外观验收、真人感终稿、作者和授权信息均未完成。衣内身体采用隐藏重叠连接；动作和穿插检查是有限采样，不代表所有动画均无问题。具体限制保存在角色状态和质量报告中。

多代理继续使用独立工作副本，只有整合者写主源。每次合并后重新确认导出和验证报告对应的新文件身份。
