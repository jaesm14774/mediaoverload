# Image Prompt Guide

## Acceptance scenario

User Given a creator submits a text2img request in a selected visual medium When MediaOverload expands the image prompt Then the final prompt preserves the requested subject, action, and style while specifying a clear focal hierarchy, purposeful use of space, useful foreground, midground, and background depth, directional light, and observable material cues without relying on generic quality slogans.

User Given a creator requests a defined cast, action, or framing with a selected visual style When MediaOverload composes the renderer prompt Then the scene contract stays authoritative, style changes only the rendering treatment, optional style composition applies only when framing is unspecified, and the prompt describes one decisive image with concrete viewpoint, light, and medium-specific surface cues.

User Given a video strategy generates a still image or opening keyframe When MediaOverload composes both prompts Then the still follows the scene-first cast, framing, light, and material contract while temporal movement stays in the video prompt.

User Given a creator requests a text2img image set When MediaOverload expands and renders the request Then it creates four to six self-contained prompts that keep the same requested subject, action, and style while changing the visual presentation enough to read as distinct compositions, and renders one image for each prompt.

User Given an image-to-video strategy creates multiple opening-frame candidates When MediaOverload prepares and renders those candidates Then each frame uses its own composition prompt for the same cast and story state, the single downstream motion prompt remains independent, and the existing review selects among the rendered frames.

User Given a video request includes a negative prompt When distinct opening-frame prompts are expanded and rendered Then every candidate retains that negative prompt alongside its own composition.

本指南提供可重複使用的構圖與提示原則。參考素材裡的「直接複製」、解析度口號與品牌／工作室風格不會自動變成 runtime 指示。creative 欄位遵守 English-only contract；解說使用繁體中文，英文範本可交給影像模型。

## 1. 使用範圍與驗收

這些機制是創作參考，不能替代目前模型的成品驗收。對實際生成結果檢查：主要機制是否可見、構圖是否可讀、角色與材質是否正確，以及是否有生成瑕疵。模型或工作流改變後須重新看圖，不沿用以前的品質分數。

## 2. 提示詞原則

### 2.0 角色小、世界大（收集資料夾實測結論）

對照「收集」資料夾與實際生成，差距主要來自五件事，皆已用 `krea2_turbo` 同 seed 對照驗證：

1. **風格池**：原本從 137 種風格等機率抽，其中含攝影、HDR、自拍、油畫人像；收集的成品全是手繪／水彩／粉彩／厚塗插畫與少量「照片＋手繪小角色」。預設風格池只保留這類（六個原有風格加 13 個插畫風格）。
2. **尺度**：舊 prompt 先寫長角色外貌，成品變成角色佔滿畫面的貼紙式特寫。收集作品是「世界是主角，角色小而簡單」，角色約佔畫面高度 1/5–1/3、偏離中心、有接觸或陰影。
3. **構圖變體**：舊變體是 eye-level、low-angle、over-the-shoulder 等相機鏡頭語言，適合照片，不適合插畫。現在改成版面概念：小角色在大世界、單一對角路徑、角色棲息在大型熟悉物件、柔邊留白小圖、窗框／門框取景、俯視桌面。
4. **負面條件無效**：`krea2_turbo` 以 `ConditioningZeroOut` 當 negative、cfg 為 1，所有 `avoid` 與 negative prompt 都不會影響畫面。限制必須用正向句寫：只出現被點名的角色、各一次；招牌、傳單、螢幕畫成抽象色塊或圖示。

5. **概念**：收集作品多半是一個可愛比喻（Kirby 躺在甜甜圈裡、排在壽司盒中），不是逐字報導。新聞場景要把角色放進故事的關鍵物件，例如躺在拿鐵杯裡、縮在外套裡、當成行動電源裡的膠囊；實測真實 LLM 在加入這條規則後，三則新聞都改寫成這種比喻構圖，之前則是「站在桌上指著招牌」的字面場景。

簡單仍然是層級清楚，不是背景空白：保留紙張肌理、前中後景、光線與少量環境線索，只在去背素材或明確要求隔離主體時使用純色底。

### 2.1 一個 visual thesis

先問：「觀眾在一秒內應該讀到哪一個關係？」例如：

- 小角色被巨大容器保護；
- 普通咖啡杯的功能失控；
- 紙飛機的軌跡畫出一個心；
- 葉片的天然葉脈支撐整個微型村落。

提示詞要把主體、主機制、情緒／結果綁在同一個畫面關係中。不要用
「可愛、夢幻、震撼、精緻」代替畫面語意。

### 2.2 把抽象情緒翻成可見物理

不要只寫 lonely、cozy、funny。改寫成：

- lonely → tiny subject inside a large cool architectural field, one localized
  warm light on the face;
- cozy → body nested inside a soft oversized container, one prop being tucked
  in, stable warm light pocket;
- funny → ordinary object performing its own function incorrectly, with a
  serious framing and one visible consequence;
- affectionate → a large simple trajectory or gesture that remains readable
  against negative space.

### 2.3 具體化瞬間，而不是物件清單

靜態 image 也需要一個 decisive visual state。使用「剛發生／正在承受／
已經造成結果」的瞬間，讓物件之間有接觸、拉力、壓縮、流動、遮蔽或
附著關係。這個方法同時適用於 I2V opening keyframe；時間軸留給 video
prompt，不要把多個後續鏡頭塞進首幀。

### 2.4 先寫幾何，再寫風格

下列關係若是笑點，就必須明寫：

- relative scale：the environment fills 95% of the frame; the hero occupies 5%;
- containment：the miniature village is cut inside one intact leaf;
- attachment：the city rests physically on the character's paws;
- layer order：six separated matte paper planes recede behind one silhouette;
- layout orientation：upper 50% photo, lower 50% drawing, not only split poster;
- cause-and-effect direction：the character pulls the claw upward; the cable bows
  and the machine loses control.

「大型、微型、剖面、雙欄」等名詞本身不夠；要寫出誰包住誰、誰支撐誰、
哪一邊在上、力往哪裡走。

### 2.5 一個 medium contract

先選一個主要媒材，再用可觀察的表面線索證明它：

- crayon：paper tooth、wax streaks、irregular charcoal outline；
- brocade charm：woven threads、embroidered piping、translucent washi；
- clay diorama：matte painted clay、layered soil、ambient occlusion；
- macro leaf art：leaf cells、natural veins、backlit translucency、edge droplets；
- documentary object comedy：glazed ceramic、wet liquid reflections、fluorescent office light。

不要同時堆 watercolor + glossy 3D + photorealistic oil + paper cut。若要
做混合媒材，必須寫清楚 transition boundary，以及每個區域保留哪一套
材料規則。

### 2.6 用留白和 focal path 控制可讀性

主笑點越簡單，越需要留白、視線路徑與尺寸層級。用物理線索帶視線：
心形軌跡、咖啡液對角線、葉脈、拱門透視、方塊剖面邊緣、布料的拉力。
細節只在主路徑讀完之後補上；不要讓每一個角落都變成同等強度的
spectacle。

### 2.7 次要細節要有工作

每個 secondary detail 至少要做到一件事：交代尺度、支持材料、放大情緒、
或完成笑點結果。例如便利袋底部的 spilled milk tea 與 rolling boba
不是裝飾，而是「擠壓逃脫已造成後果」的證據。無法支持主讀的細節刪除。

### 2.8 Negative constraints 要針對失敗模式

泛用的 high quality 不如具體的：

- no duplicate protagonist；
- no extra characters unless multiplicity is the declared mechanism；
- no readable pseudo-text；
- no disconnected floating cutouts；
- no material outside the declared medium；
- no layout inversion；
- no static setup when the prompt's joke is a reversal。

Avoid list 應短而有因果，避免塞滿互相矛盾的風格形容詞。

## 3. 可直接套用的 English prompt formula

~~~text
Use case: <illustration-story|stylized-concept|photorealistic-natural|ads-marketing>
Asset type: <final still|opening keyframe|poster|diorama|character design sheet>
Primary request: <one visual thesis in one sentence>
Scene/backdrop: <where the scene is and what establishes scale>
Subject: <identity, silhouette, expression, and role>
Action or visual state: <one decisive visible moment and its physical consequence>
Environment: <one dominant force, prop, architecture, or contained world>
Composition/framing: <orientation, relative scale, focal path, and negative space>
Lighting/mood: <localized light versus broad light; emotion shown physically>
Color palette: <controlled palette with one accent hierarchy>
Materials/textures: <one medium plus observable surface cues>
Constraints: <count, attachment, containment, layer order, exact layout, text policy>
Avoid: <short list of likely visual failures>
~~~

對靜態圖，Action or visual state 可以是靜止姿態，但仍要交代它和主機制
的關係。對 image-to-video，這一欄只描述首幀的動作 onset；後續
anticipation、contact、reaction、settle 應寫入 video prompt。

## 4. 已通過案例的可泛化配方

下列配方可泛化，不複製某一張圖的角色、標誌或品牌。

### A. Craft object + miniature center

~~~text
Make one ordinary craft object extraordinary through its silhouette first:
a three-dimensional flower-bud charm, not a flat pouch. Place one tiny subject
inside the center so scale is undeniable. Use backlighting through a translucent
inner layer to reveal construction. Anchor the fantasy with real craft hardware
such as a braided cord and bell. Name woven thread, embroidery, paper fiber, and
dew only after the object structure is clear.
~~~

保留的是：輪廓新穎、內外層關係、中心尺度錨點、可觸摸材料；不是固定
牡丹、特定角色或祈文。

### B. Minimal gesture + oversized negative-space shape

~~~text
Keep the thrower tiny in the lower edge of a vertical picture-book frame. Make
the trajectory itself the main subject: a paper plane leaves a dotted heart
path that is already visibly halfway through its loop. Use a quiet sunset wash,
few clouds, generous negative space, and one clear direction from the thrower to
the plane. The path must be a graphic shape, not a field of random floating
icons.
~~~

保留的是：大圖形軌跡、微小行為者、留白與單一方向；若文字不是重點，
不要依賴紙飛機上的字來完成情緒。

### C. Serious framing + one ordinary function failure

~~~text
Photograph one ordinary office object as if it were a serious corporate
documentary. Give it one readable expressive face, then make its native function
fail physically in one direction: liquid overflows, spreads, and reaches nearby
surfaces. Keep the office lighting and depth of field believable so the cartoon
expression is the only absurd layer. The failure must be visible in the object
and its immediate consequence, not added as unrelated explosions.
~~~

保留的是：正經攝影語法與單一功能失控的衝突；不要把辦公室變成第二個
主角，也不要用很多同時爆炸的小笑點。

### D. Declared multiplicity + pressure evidence

~~~text
Declare multiplicity as the joke, then cap the count. One translucent container
is stretched nearly to its limit; the main face is compressed at the opening,
while a small number of copies perform different micro-actions. Show pressure
through stretched plastic, a side tear, bent handles, facial compression, and
ground-level spill debris. Keep the main face dominant and make every copy's
pose distinct enough to read.
~~~

保留的是：容器壓力、有限複數、主次層級、地面後果；不能把「很多角色」
當成預設配置，必須由 request 明確啟用。

### E. Coherent cutaway + value-separated zones

~~~text
Use one clean floating cube as the only world boundary. State the surface level,
the geological cutaway, and each underground chamber. Separate each zone by
material and light temperature while keeping all props physically supported by
the cube. Use isometric framing so the viewer can discover the top world and the
hidden world in one glance. Every detail must belong to a specific layer.
~~~

保留的是：一個容器邊界、地質連接、區域色溫／材質區隔、可探索但不散亂；
不是固定海島、寶箱或水晶。

### F. Host material as the credibility anchor

~~~text
Preserve the complete outer silhouette of one natural host object. Carve a
miniature narrative into it, and route every fine element back into a visible
structural vein or support. Use transmitted light, surface cells, and edge
droplets to prove thin organic material at macro scale. Reject any isolated
floating fragment or a second host object.
~~~

保留的是：宿主輪廓、結構支撐、背光證據與微距尺度；不要把葉雕寫成
一般 3D 小模型貼在葉片上。

### G. Design sheet with restrained callouts

~~~text
Center one hero silhouette in a restrained design-sheet layout. Use a small
number of peripheral callout boxes to prove fabric, ornament, palette, and
construction logic. Connect callouts with thin, consistent guide lines. Let one
proportion mismatch create the charm, while keeping the main figure readable.
Treat the callouts as visual swatches and close-ups unless exact text is
explicitly required.
~~~

保留的是：主輪廓、少量 callout、導線系統與比例反差；不要讓 callout
變成一張文字密集的資訊圖。

## 5. 防錯規則

這些規則不是負面評分清單，而是下一輪 prompt 的早期檢查：

- 反向劫持若是笑點，首幀必須同時顯示 opposing vectors、
  cable slack/strain、機器失控方向；「被夾起來」不是「帶走夾爪」。
- split poster 必須明寫 upper/lower 或 left/right、各自比例、
  共用姿態與 crop；layout orientation 錯了就是機制失敗。
- 若宣稱 strict paper-cut，角色、燈具、景物都必須在同一種
  matte paper vocabulary；金屬、玻璃、立體塑膠會破壞材料 contract。
- 文化職能若重要，要放進可辨識道具與動作；宏偉背景不能代替
  guardian role。
- 若需要 warm/cool tension，暖光要真的落在主體，不只是背景
  有一個粉紅色角色。
- 測量笑點依賴儀器規格時，要讓量測系統本身可辨識；若不需要
  文字，就改用明確 gauge、needle、droplet count 等無字視覺證據。
- 95/5 scale joke 要真的接近 95/5；宏偉背景不能只寫在描述，
  必須指定主體佔比、位置與可讀對比。
- sketch-to-reality 的「忠實」包含錯誤比例與笨拙受力，不能
  把塗鴉美化成普通可愛模型。

## 6. Runtime integration

`krea2_turbo` 工作流只接收文字提示，沒有 conditioning image 輸入；因此 runtime 把範本中的畫面主張、角色關係、視角、光線與媒材線索轉成提示規則。場景與選角先建立，風格只處理渲染；風格內的構圖提示僅在請求沒有指定構圖時採用。

- `agentic/src/agentic/runtime/prompting.py`
  - 定義 `IMAGE_PROMPT_CONTRACT` 與 `IMAGE_PROMPT_QUALITY_CLAUSE`。
  - `build_goal_brief()` 在 `image` 的 template fallback 加入畫面品質要求，並將參考風格附在場景描述之後。
- `agentic/src/agentic/runtime/prompt_tasks/scenes.py`
  - `expand_goal()` 與 `compose_prompt()` 的 image 請求都使用相同 contract；影片請求只把此 contract 套用到 opening keyframe，時間動作仍留在 video prompt。必要角色會被提示在同一畫面形成可見互動，風格不能新增角色或道具。
  - LLM 產生的 image prompt 與 `expand_goal()` 的 template fallback 都會附上精簡品質要求。
- 一般 text2img 會先展開 4–6 份完整 prompt，每份保留相同題材、角色和動作，改變構圖、視角、景深或焦點路徑，再由 `media.image.render_batch` 每份渲染一張。每個變體會檢查必須出現的主體名稱；LLM 不可用時使用六種明確構圖方向的模板。
- Text2Image2Video、啟用首幀審核的 Text2Video，以及 Native H3 Story／FL2VA 在產生多張首幀候選時，會用同一個 `agent.image.prompt_variants` 為候選各寫一份構圖 prompt，再交給 `media.image.render_batch`。變體維持相同角色、場景與凍結起始狀態；既有影片 prompt 繼續單獨描述後續動作。審核重試會沿用這個首幀流程並套用審核意見。
- Native H3 L2VA 的文字生圖圖像是末幀生成的身份來源，Ref2VA 的自動生圖圖像是參考素材；兩者不當成影片首幀候選處理。
- `agentic/src/agentic/skills/comfy_image.py`、`agentic/src/agentic/skills/agent_primitives.py` 與 `agentic/src/agentic/skills/motion_comic.py`
  - 所有 runtime 的文字生圖入口（含影片首幀、原生 H3 keyframe、motion-comic panels 與 story-card 背景）會把精簡品質要求附在實際靜態圖 prompt；品質要求不強迫三層景深或單一光源，`sticker_pack`、`animated_sticker`、`game_sprite` 保留各自的素材規格，影片 motion prompt 維持獨立。
- `agentic/src/agentic/runtime/illustration_style.py`
  - 一般圖片不再無條件追加通用 storybook 風格段落；只有貼圖隔離主體保留其去背輪廓與背景規格。媒材與情緒由當次 brief 和選定風格決定。

提示規則讓畫面要求更具體，但成品仍需實際生成並人工檢視；不以提示詞本身宣稱視覺品質已達標。

## 7. 每次生成前後的最小檢查

### Before generation

1. 一句話寫出 visual thesis。
2. 指定主體數量與是否允許複數。
3. 指定主機制的物理關係：壓縮、附著、穿透、流動、支撐、軌跡或色溫。
4. 指定畫面方向、相對尺寸、focal path 與留白。
5. 選一個媒材，列出 3–5 個可觀察表面線索。
6. 對文字、圖表、多欄、層數與 exact copy 寫出明確 contract。

### After generation

1. 不看 prompt，先用縮圖判斷主體與主機制是否一秒可讀。
2. 再放大檢查材料是否一致、幾何是否閉合、是否有 duplicate／pseudo-text。
3. 逐項對照 constraints；缺少核心機制就淘汰，不用「很漂亮」補分。
4. 需要 retry 時只改最弱的一個 dimension，保留已成功的 style signature。
5. 若進入 90+ gate，保留 prompt、PNG、各 reviewer 分數與採用理由；
   未通過者不得寫入 guide。
