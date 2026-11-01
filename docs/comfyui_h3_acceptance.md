# ComfyUI H3 驗收契約

- User Given 既有角色生成策略與設定 When 建立執行計畫 Then 文生影、圖生影、首尾幀、尾幀、參考素材、長影片與動態貼圖都使用同一個 ComfyUI 服務，保留原本的節點依賴、審查、QA、編輯與發布政策。
- User Given 完整的 H3 場景描述 When 從生成工具入口提交 Then 原始 prompt、seed、步數、長寬、影片長度與模型 profile 傳入 ComfyUI；不加入提示詞增強或其他生成引擎。
- User Given 文字、首圖、首尾圖、尾圖或參考素材 When 渲染對應 H3 模式 Then 各模式只接收其允許的條件，缺少必要素材或有無效 prompt 時在 GPU 提交前拒絕。
- User Given 已審查的圖片或影片參考素材 When 執行 Ref2VA Then 保留素材順序與角色，使用 Ref2VA 模型，且不傳入參考音軌。
- User Given 需要檢查完整性的 H3 參考影片 When MediaOverload 驗證可讀性與幀數 Then 單次完整影音解碼回報精確影片幀數，且素材雜湊與原本錯誤邊界保留。
- User Given 大型影片參考素材或 ComfyUI 影片成品 When MediaOverload 上傳或保存該媒體 Then 它以 1 MiB 有界記憶體區塊串流傳輸，保存完整檔案，且失敗的下載不留下可被後續節點使用的半成品。
- User Given 一張 conditioning 圖要產生多個 img2img 候選 When MediaOverload 逐個 seed 生成候選 Then 它只上傳來源圖一次、重用同一個 ComfyUI 檔名，並保留每個 seed 的獨立成品。
- User Given 一支含音訊的生成影片 When MediaOverload 執行 video QA Then 它回報一致的影片時長、音訊時長、聲道、音量及靜音分析結果。
- User Given 一支畫布尺寸與目標相同且影片／音訊串流可直接封裝的 MP4 When MediaOverload 正規化畫布 Then 成品保留影格、FPS、時長與音訊且可正常播放。
- User Given ComfyUI 原始 MP4 已符合 H3 交付的畫布、幀數、FPS 與雙聲道條件 When MediaOverload 直接複製原檔交付 Then 它沿用已取得的原檔探測結果，不重複完整計幀，且交付契約維持成立。
- User Given 有效 H3 生成參數 When 渲染成功 Then 交付可解碼 MP4、請求的長寬／幀數／24 FPS 與原生雙聲道音訊，並保留請求與成品紀錄。
- User Given 尚未安裝模型或節點的 ComfyUI When 執行 Then 回報缺少的資產／節點，不切換模型、不回傳假成品。
- User Given 已建立的 H3 workflow graph When 預檢 ComfyUI 節點 Then 只查詢 graph 使用的唯一節點型別，並完整列出缺少的必要節點。
- User Given 擴充同類模型的需求 When 加入工作流與候選設定 Then 沿用既有策略、技能及 ComfyUI 服務契約，不新增另一套驅動軟體。

本機測試驗證真實工作流、計畫與輸入／交付契約；受控 GPU 驗收另列實際送出的 ComfyUI 請求、history 與 ffprobe 證據。LLM、Discord、社群發布及自然排程可靠度需各自的外部驗收。
