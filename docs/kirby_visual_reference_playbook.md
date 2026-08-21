# Kirby visual reference playbook

這份筆記只整理 `C:\Users\jaesm14774\Downloads\收集\` 中的使用者提供素材，並把可重複的創作機制轉成 repo 內的 storyboard、style、workflow、prompt 與 QA 規則。

## 指示與素材的邊界

收集資料夾目前只有 8 張 PNG 與 5 部 MP4，沒有獨立的 Markdown、TXT、DOCX 或 PDF 指示文件。因此，社群影片畫面中的帳號列、追蹤按鈕、浮水印、平台 UI、貼文文字與音訊標記，都只視為來源包裝，不視為本專案要複製的創作指令。真正的工作要求是：逐張、逐部觀察後，將可取的可愛機制轉成現有 Kirby pipeline 的改善。

## 逐張觀察

### 圖片

1. `螢幕擷取畫面 2026-08-17 230833.png`：霧紫紙張顆粒、稀疏星火與大量留白；小角色躲在畫面底部，情緒不是靠複雜表情，而是靠「小小的身體面對很大的天空」。可取機制是低刺激、手作材質、孤單但安全的觀看姿態。
2. `螢幕擷取畫面 2026-08-17 230855.png`：彩色傘海幾乎填滿上半部，角色被放在低角度前景；色彩重複但每把傘仍有清楚形狀，形成歡樂的規模反差。可取機制是單一環境 gimmick 支撐整張圖，不需要額外劇情裝置。
3. `螢幕擷取畫面 2026-08-17 230944.png`：圓潤 Kirby 坐在巨大發光星體上，臉卻是認真的「可愛攻擊性」表情；黃色光、紅橙外圈與星芒把一個情緒放大成主視覺。可取機制是可愛身體與強烈決心的衝突，以及一個巨大道具承擔情緒。
4. `螢幕擷取畫面 2026-08-17 231003.png`：廚師 Kirby 在煎鍋旁完成拋接，半空中的食物、鍋柄斜線、臉部朝向形成很清楚的動作路徑。可取機制是 anticipation → 飛行 → 接住／落點，觀眾不用文字就知道下一步。
5. `螢幕擷取畫面 2026-08-17 231107.png`：小角色坐在巨大圓形平台或容器中心，周圍是綠金色流動水面與拉出的白色水線；角色很小，環境很大，但焦點仍集中在中央。可取機制是「微小主角在大環境中維持可讀姿態」，以及一條可追蹤的物理運動線。
6. `螢幕擷取畫面 2026-08-18 222218.png`：角色被拉成極端長條，眼睛與臉仍清楚；背景簡單到只剩柔和粉色與速度線。可取機制是純粹的 squash-and-stretch，變形本身就是笑點，不能被背景細節搶走。
7. `螢幕擷取畫面 2026-08-18 222648.png`：小 Kirby 躺在巨大金屬杯／容器裡，藍色夜景與星光包住主體，前景有暖色小燈。可取機制是容器變成安全窩，利用「巨大物件 = 舒適空間」做 cozy scale contrast。
8. `螢幕擷取畫面 2026-08-18 222757.png`：三個簡化的角色在底部同步仰望煙火；上方事件複雜、下方角色極簡，三張臉只保留不同的小反應。可取機制是共同視線、上大下小的構圖，以及讓背景事件替角色承擔 spectacle。

## 逐部觀察

### 影片

1. `錄製內容 2026-08-17 230920.mp4`（約 5.53 秒）：以多個料理／盤面快速切換，角色每次用張嘴、閉眼或誇張笑容回應食物。可取的是「一個主題、連續幾個具體道具、反應做節拍」；不應直接搬成 H3 的多段 lore 蒙太奇，但很適合 carousel、reference bundle 或料理主題的候選素材。
2. `錄製內容 2026-08-17 231055.mp4`（7 秒）：先展示角色化的咖哩盤，再讓廚師角色試吃與互動，最後用亮色轉場落到實拍料理成品。可取的是 in-world preparation → emotional reveal → real-world artifact reveal；若用於本專案，應把最後的「成品」改成同一個無文字的物理 payoff。
3. `錄製內容 2026-08-17 231130.mp4`（5 秒）：三個幽靈型角色在圓盤／圓形構圖中持續浮動、煙霧擴散、表情微妙變化；重點是群像海報感與可重播氛圍，不是完整因果故事。可取的是固定主構圖、角色群的同步微動與單一 FX 層。
4. `錄製內容 2026-08-18 222720.mp4`（5.1 秒）：白底中央只有 Kirby，透過非常細小的上下呼吸／身體擺動完成 loop。可取的是極簡背景、單一情緒、首尾近似、低成本但高辨識度的 loop；這是 animated-sticker prompt 最直接的參考。
5. `錄製內容 2026-08-18 222856.mp4`（約 9.07 秒）：真實手、筷子、海苔與盤子和 2D 小貓直接互動；食物或海苔壓到小貓後，小貓扁掉、哭、再回彈並重新被吸引。可取的是明確的 cause → contact → reaction → settle，以及實拍物件與動畫角色之間的比例笑點。

## 已吸收的共同機制

- 一支短片只需要一個 dominant prop 或環境力量；道具要真的造成角色的反應。
- 先給可讀的 anticipation，再給接觸／撞擊／拉伸，再給臉或身體反應，最後讓結果落定。
- 小主角對大容器、大星體、大水面、大傘海的比例反差，比堆疊世界觀更容易產生可愛感。
- 變形不是裝飾：squash、stretch、wobble、recoil 只有在畫面中的力造成它時才成立。
- 背景要服務主動作：留白、單色或單一大場景都可以；不要同時加入多組 spectacle。
- loop 可以首尾呼應，但不能因為回到首幀而抹掉 payoff；最後一格要仍看得出「什麼已經改變」。
- 社群 UI、文字與浮水印不進 prompt；料理、杯子、煙火、星體、傘海等只抽象成物理機制與構圖語言。

## Repo 內的落地位置

| 層 | 落地內容 |
| --- | --- |
| Kirby config | `generation.visual_style_contract` 增加 tactile pastel、scale contrast、單一 palette、可回放 ending；creative brief 加入 prop-caused reaction 與 settled payoff。 |
| Native storyboard | `kirby_native_15s.yaml`、`kirby_native_15s_5beat.yaml`、`kirby_native_20s.yaml` 都要求 dominant prop／force、可讀反應與 loop echo。 |
| H3 prompt | `compose_minimax_h3_prompt()` 加入 cute physical-comedy、scale-and-silhouette、replay 三個 contract。 |
| 一般短片與貼圖 prompt | 5–6 秒改成單一完整 physical action；animated sticker 改成 anticipation → impact → settle，鎖定鏡頭與首尾可接。 |
| Routing | 5–9 秒、食物／桌面互動、反應 loop、誇張伸縮、小角色對大物件等需求優先走 `text2image2video` 的首幀審核 + I2V，而不是 `text2longvideo`。 |
| Semantic QA | 將小對大、觸感互動、變形／反彈、留白與 opening echo 設為正向品質訊號，但仍以可見 cause-and-effect、action completion、payoff 為硬證據。 |

## 使用判準

5–9 秒的單一可愛動作：優先 `text2image2video` 或 animated-sticker 類路線。需要完整三拍因果、新聞 anchor、可人工審核的開場：才使用 native H3 15 秒路線。需要多張既有素材共同保留身份／質感／運鏡：才使用 Ref2VA，並保留人工 reference gate。

這些規則是從素材中抽出的創作機制，不是複製任何單一作者的角色、畫面或浮水印。
