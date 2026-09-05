# Image Prompt Guide

本指南把使用者提供的 Complete_39_AI_Art_Prompt_Master_Library.md 轉成
MediaOverload 可重複使用的提示詞規則。原始文件是研究素材；文件內的
「直接複製」、「8K」、「某品牌／工作室風格」等文字，不會自動變成 repo
runtime 指示。creative 欄位仍遵守 repo 的 English-only contract；本頁的
解說以繁體中文撰寫，英文範本則可直接交給影像模型。

## 1. 研究範圍與採用門檻

39 組提示詞已全部閱讀並按創作機制編目。實際生成 18 次代表性候選：
16 張成功出圖、2 張在影像服務安全階段被拒絕（原始條目 20、30），因此
那 2 張沒有品質分數。

每張成功候選由三個不同高規模型獨立檢視實際 PNG：

- gpt-5.6-sol
- gpt-5.6-terra
- gpt-5.5

五項各 20 分：機制符合度、構圖可讀性、意境與魅力、材質／渲染控制、
生成潔淨度。採用條件是所有 reviewer 的總分都至少 90，且沒有 critical
failure；平均分不能救回最低分低於 90 的候選。

通過的原始條目只有 1、5、9、14、18、25、28。完整分數與淘汰理由見
output/prompt_library_eval/2026-09-01/review_report.md；候選與來源對照
見 output/prompt_library_eval/2026-09-01/evaluation_manifest.md。

## 2. 39 組素材的機制地圖

| 條目 | 研究到的核心機制 | 使用時的硬條件 |
| --- | --- | --- |
| 1–5 東方美學與節慶 | 服裝混合、文化閘門、節慶競速、墨色變形、花朵形御守 | 先鎖物件輪廓、材料與色溫分區；文化角色需要可辨識的職能道具，不靠抽象標籤 |
| 6–12 Q版與療癒 | 蠟筆堡壘、光雲擁抱、雨窗孤獨、心形軌跡、UI 隱喻、極簡線條、角色堆疊 | 觸感材質、單一情緒、留白與可讀表情優先；多角色只在「壓縮／堆疊」本身是笑點時啟用 |
| 13–19 荒謬日常 | 正經上班反差、物件功能失控、微小焦慮災難化、情緒量測、誤解四格、袋中擠壓、自戀購物 | 讓一個普通物件承擔一個失控機制；四格或圖表需先指定版面方向與比例；文字不是可靠的笑點載體 |
| 20–24 變身與史詩 | 能力變身、三段因果、巨型地標睡姿、節慶騎乘、史前競速混亂 | 變身必須保留基礎輪廓並說清「前狀態 → 轉換 → 後狀態」；史詩背景不能吃掉可愛主體 |
| 25–29 微縮與立體 | 浮空剖面方塊、附著城市、盲盒工廠、葉脈葉雕、球體瑜伽 | 明確指定 containment、attachment、layer order 與支撐關係；每個小區域要有不同 value/material zone |
| 30–34 動態短影音 | DJ 突變、夾爪反向劫持、塗鴉實體化、同步舞、天空翻滾 | 靜態首幀只凍結最能證明機制的瞬間；反轉需要 opposing vectors、受力／位移與結果同時可見 |
| 35–39 海報與排版 | 四季／烘焙日曆、上下雙媒材海報、版畫、紙雕燈箱 | 版面方向、分割比例、文字是否必要、層數與同一材料系統都要明寫；layout geometry 是內容，不是裝飾 |

這張地圖是「每個 prompt 都有被理解」的索引，不代表 39 組都已通過
成品 gate。只有有獨立分數證據的條目才可寫入「已驗證技巧」。

## 3. 已驗證的提示詞原則

### 3.1 一個 visual thesis

先問：「觀眾在一秒內應該讀到哪一個關係？」例如：

- 小角色被巨大容器保護；
- 普通咖啡杯的功能失控；
- 紙飛機的軌跡畫出一個心；
- 葉片的天然葉脈支撐整個微型村落。

提示詞要把主體、主機制、情緒／結果綁在同一個畫面關係中。不要用
「可愛、夢幻、震撼、精緻」代替畫面語意。

### 3.2 把抽象情緒翻成可見物理

不要只寫 lonely、cozy、funny。改寫成：

- lonely → tiny subject inside a large cool architectural field, one localized
  warm light on the face;
- cozy → body nested inside a soft oversized container, one prop being tucked
  in, stable warm light pocket;
- funny → ordinary object performing its own function incorrectly, with a
  serious framing and one visible consequence;
- affectionate → a large simple trajectory or gesture that remains readable
  against negative space.

### 3.3 具體化瞬間，而不是物件清單

靜態 image 也需要一個 decisive visual state。使用「剛發生／正在承受／
已經造成結果」的瞬間，讓物件之間有接觸、拉力、壓縮、流動、遮蔽或
附著關係。這個方法同時適用於 I2V opening keyframe；時間軸留給 video
prompt，不要把多個後續鏡頭塞進首幀。

### 3.4 先寫幾何，再寫風格

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

### 3.5 一個 medium contract

先選一個主要媒材，再用可觀察的表面線索證明它：

- crayon：paper tooth、wax streaks、irregular charcoal outline；
- brocade charm：woven threads、embroidered piping、translucent washi；
- clay diorama：matte painted clay、layered soil、ambient occlusion；
- macro leaf art：leaf cells、natural veins、backlit translucency、edge droplets；
- documentary object comedy：glazed ceramic、wet liquid reflections、fluorescent office light。

不要同時堆 watercolor + glossy 3D + photorealistic oil + paper cut。若要
做混合媒材，必須寫清楚 transition boundary，以及每個區域保留哪一套
材料規則。

### 3.6 用留白和 focal path 控制可讀性

主笑點越簡單，越需要留白、視線路徑與尺寸層級。用物理線索帶視線：
心形軌跡、咖啡液對角線、葉脈、拱門透視、方塊剖面邊緣、布料的拉力。
細節只在主路徑讀完之後補上；不要讓每一個角落都變成同等強度的
spectacle。

### 3.7 次要細節要有工作

每個 secondary detail 至少要做到一件事：交代尺度、支持材料、放大情緒、
或完成笑點結果。例如便利袋底部的 spilled milk tea 與 rolling boba
不是裝飾，而是「擠壓逃脫已造成後果」的證據。無法支持主讀的細節刪除。

### 3.8 Negative constraints 要針對失敗模式

泛用的 high quality 不如具體的：

- no duplicate protagonist；
- no extra characters unless multiplicity is the declared mechanism；
- no readable pseudo-text；
- no disconnected floating cutouts；
- no material outside the declared medium；
- no layout inversion；
- no static setup when the prompt's joke is a reversal。

Avoid list 應短而有因果，避免塞滿互相矛盾的風格形容詞。

## 4. 可直接套用的 English prompt formula

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

## 5. 已通過案例的可泛化配方

下列是只從 strict 90+ accepted set 提取的配方，不是複製某一張圖的
角色、標誌或品牌。

### A. Craft object + miniature center

來源條目 5，三位 reviewer 分數為 95、92、94，平均 94，最低 92。

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

來源條目 9，分數為 95、94、94，平均 94，最低 94。

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

來源條目 14，分數為 93、90、91，平均 91，最低 90。

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

來源條目 18，分數為 93、92、94，平均 93，最低 92。

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

來源條目 25，分數為 96、90、95，平均 94，最低 90。

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

來源條目 28，分數為 96、91、92，平均 93，最低 91。

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

來源條目 1，分數為 96、93、93，平均 94，最低 93。

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

## 6. 淘汰案例轉成的防錯規則

這些規則不是負面評分清單，而是下一輪 prompt 的早期檢查：

- 條目 31：反向劫持若是笑點，首幀必須同時顯示 opposing vectors、
  cable slack/strain、機器失控方向；「被夾起來」不是「帶走夾爪」。
- 條目 37：split poster 必須明寫 upper/lower 或 left/right、各自比例、
  共用姿態與 crop；layout orientation 錯了就是機制失敗。
- 條目 39：若宣稱 strict paper-cut，角色、燈具、景物都必須在同一種
  matte paper vocabulary；金屬、玻璃、立體塑膠會破壞材料 contract。
- 條目 2：文化職能若重要，要放進可辨識道具與動作；宏偉背景不能代替
  guardian role。
- 條目 8：若需要 warm/cool tension，暖光要真的落在主體，不只是背景
  有一個粉紅色角色。
- 條目 16：測量笑點依賴儀器規格時，要讓量測系統本身可辨識；若不需要
  文字，就改用明確 gauge、needle、droplet count 等無字視覺證據。
- 條目 22：95/5 scale joke 要真的接近 95/5；宏偉背景不能只寫在描述，
  必須指定主體佔比、位置與可讀對比。
- 條目 32：sketch-to-reality 的「忠實」包含錯誤比例與笨拙受力，不能
  把塗鴉美化成普通可愛模型。

## 7. Repo 落地

這次只做窄幅、可驗證的 prompt contract 更新：

- agentic/src/agentic/runtime/prompting.py
  - 新增 IMAGE_PROMPT_CONTRACT，供 LLM 生成 image prompt 與 opening
    keyframe 時使用。
  - image 類型的 fallback action 改成 decisive visual state；quality
    clause 加入單一主機制、留白與可觀察材質要求。
- agentic/src/agentic/runtime/llm_engine.py
  - expand_goal() 與 compose_prompt() 都注入同一份 image contract，
    避免 LLM route 和 fallback route 的學習規則分裂。
- configs/characters/kirby.yaml
  - 在既有 generation.visual_style_contract 中加入 still/opening
    keyframe 的 visual thesis、scale/attachment/layer geometry 與 material
    cues；沒有寫死新角色或新 provider。
- output/prompt_library_eval/2026-09-01/accepted/
  - 只保留七個 strict 90+ 案例的 copied PNG；被淘汰候選未進入 repo
    accepted folder，也未進 guide。

這些改動沒有移除既有 Ollama／OpenRouter 設定，也沒有改變角色選擇、
社群發布或 video temporal contract。Image guide 是新增的可讀文件，
runtime 只吸收已驗證的泛化規則。

## 8. 每次生成前後的最小檢查

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
