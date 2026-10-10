# ComfyUI H3 驗收契約

- User Given 既有角色生成策略與設定 When 建立執行計畫 Then 文生影、圖生影、首尾幀、尾幀、參考素材、長影片與動態貼圖都使用同一個 ComfyUI 服務，保留原本的節點依賴、審查、QA、編輯與發布政策。
- User Given 完整的 H3 場景描述 When 從生成工具入口提交 Then 原始 prompt、seed、步數、長寬、影片長度與模型 profile 傳入 ComfyUI；不加入提示詞增強或其他生成引擎。
- User Given 文字、首圖、首尾圖、尾圖或參考素材 When 渲染對應 H3 模式 Then 各模式只接收其允許的條件，缺少必要素材或有無效 prompt 時在 GPU 提交前拒絕。
- User Given 已審查的圖片或影片參考素材 When 執行 Ref2VA Then 保留素材順序與角色，使用 Ref2VA 模型，且不傳入參考音軌。
- User Given 有效 H3 生成參數 When 渲染成功 Then 交付可解碼 MP4、請求的長寬／幀數／24 FPS 與原生雙聲道音訊，並保留請求與成品紀錄。
- User Given 尚未安裝模型或節點的 ComfyUI When 執行 Then 回報缺少的資產／節點，不切換模型、不回傳假成品。
- User Given 擴充同類模型的需求 When 加入工作流與候選設定 Then 沿用既有策略、技能及 ComfyUI 服務契約，不新增另一套驅動軟體。

本機測試驗證真實工作流、計畫與輸入／交付契約；受控 GPU 驗收另列實際送出的 ComfyUI 請求、history 與 ffprobe 證據。LLM、Discord、社群發布及自然排程可靠度需各自的外部驗收。
