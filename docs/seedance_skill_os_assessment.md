# Seedance 2.0 Skill OS 評估與 MediaOverload 整合紀錄

日期：2026-10-23
上游快照：[`Emily2040/seedance-2.0`](https://github.com/Emily2040/seedance-2.0/tree/4668457e560eee06e95d7fcfdf441c8c0bba802e)，v6.7.0，commit `4668457e560eee06e95d7fcfdf441c8c0bba802e`。

## 結論

Seedance 2.0 值得參考的是把創作指引、連續性狀態、驗證器與評估流程組成一套可追溯的 Skill OS。它目前沒有已完成的模型生成評估或渲染試驗，因此不能據此判斷 Seedance 的生成品質優於 MediaOverload 的 H3 流程。

本次只整合一項符合 MediaOverload 流程的設計：啟用 `longvideo_review_policy: every_segment` 且要求人工審核時，每段影片須先由人接受，該段觀察到的尾幀才可提供下一段延續。原有 H3 單鏡頭分段、尾幀鏈結和最終成片審核仍各司其職；Seedance 的多鏡頭 prompt 表格沒有套用到 H3 單段影片。

## 評估方法與證據

上游使用隔離的淺層 clone 檢查，沒有把其程式碼或依賴複製進 MediaOverload。閱讀 README、根目錄 `SKILL.md`、驗證指南、序列狀態及分鏡參考、評估紀錄，並執行上游提供的 `python scripts/validate_repo.py`。

主要上游來源：

- [README 與驗證範圍](https://github.com/Emily2040/seedance-2.0/blob/4668457e560eee06e95d7fcfdf441c8c0bba802e/README.md)
- [根目錄 Skill](https://github.com/Emily2040/seedance-2.0/blob/4668457e560eee06e95d7fcfdf441c8c0bba802e/SKILL.md)
- [Validation guide](https://github.com/Emily2040/seedance-2.0/blob/4668457e560eee06e95d7fcfdf441c8c0bba802e/docs/VALIDATION.md)
- [Sequence project state](https://github.com/Emily2040/seedance-2.0/blob/4668457e560eee06e95d7fcfdf441c8c0bba802e/references/sequence-project-state.md)
- [Shot table](https://github.com/Emily2040/seedance-2.0/blob/4668457e560eee06e95d7fcfdf441c8c0bba802e/references/shot-table.md)
- [Rendered pilot plan](https://github.com/Emily2040/seedance-2.0/blob/4668457e560eee06e95d7fcfdf441c8c0bba802e/evals/capped-rendered-pilot.md) 與 [evaluation ledger](https://github.com/Emily2040/seedance-2.0/blob/4668457e560eee06e95d7fcfdf441c8c0bba802e/evals/eval-run-ledger.md)

## 上游設計中可移植的部分

1. **把不同種類的保證分開。** 結構檢查、離線契約、人工創作判斷、模型生成與實際渲染是不同證據。上游文件明確提醒，通過 repo validator 不代表影片生成或視覺品質良好。
2. **序列狀態只接受已接受的實際結果。** 上游狀態文件區分已確認的參考與實際片段尾端；被拒絕的生成結果不應成為下一段的父狀態，場景切換則重新錨定狀態。
3. **可檢驗的工作流契約。** 它提供 prompt lint、序列狀態驗證、行為契約、eval harness 等離線檢查，讓規則可被機器驗證。
4. **分鏡表先於長 prompt。** 每鏡頭先定動作、鏡頭與銜接資訊，再編譯成模型 prompt；這適用多鏡頭生成任務，不等同於 H3 連續單鏡頭片段的最佳表達方式。

## 上游實際測試結果

`python scripts/validate_repo.py` 的步驟 1–17 通過，包括 28 個子技能的結構、151 筆 eval case schema、JSON schema、序列狀態及 47 筆 sequence eval、prompt lint、eval harness 與 frame extraction 自我檢查。它接著啟動總計 1,382 項的單元測試；Windows 安裝回歸耗時很長，完整 validator 在這一段執行約 25 分鐘後中止，故不能宣稱全套通過。

Windows 隔離重跑結果：兩個 masthead 測試 `test_masthead_recovery_is_cwd_independent` 和 `test_installed_modules_must_match_the_locked_wheel_bytes` 在將 `TEMP`/`TMP` 固定為使用者目錄長路徑後通過；初次 validator 中出現的路徑差異可由臨時目錄短路徑表示差異重現。`test_kill_in_promotion_gap_restores_previous_install_on_retry` 單獨重跑仍失敗：受控安裝程序在 30 秒內未觸及測試暫停點，且沒有輸出。這是上游安裝測試在本 Windows 環境的未解結果，與模型生成或 prompt stress 分數無關。未授權 POSIX 的測試依平台略過，Windows symlink 測試亦因權限不足略過。

上游內部 prompt architecture stress 評估使用 102 個 prompts，三組各 34 個。`skill_formula` 平均 3.89 並通過其結構門檻；`quickstart_style` 平均 3.13、`naive_online` 平均 2.46，兩組未通過同一門檻。這支持「此 repo 的 Skill Formula 在它自己的結構評分規則下較完整」，不測量 Seedance 模型的影片畫質、指令遵循或 MediaOverload 的成片品質。

README 和評估紀錄指出 model-in-loop 分數及 capped rendered pilot 尚未完成；首頁展示片段也標成 `unrendered`。所以本次沒有 Seedance 實際生成影片可作 A/B 比較，也沒有證據能宣稱上游 repo 的影片表現比較好。

## MediaOverload 的差距與整合

本地長片規劃器已經逐段生成、技術檢查影片，並從實際片段抽尾幀給下一段。不過，原有 `every_segment` 設定只會要求逐段素材審核；影片本身可以在未被人接受前先供下一段延續。

現在當 `every_segment` 與 `require_human_review` 同時生效時，規劃器在每段技術 QA 後加入影片接受審核。接受審核的 `media_paths` 才提供給尾幀抽取，且每段接受節點都是最終合片的前置條件；拒絕、審核不可用或等待失敗會阻擋序列。最後成片的審核仍是另一個決策。這次沿用現有 Discord review、執行圖和 policy，不新增依賴或相容分支。

先新增 BDD 驗收案例並確認它在修改前失敗；整合後，三個相關案例通過，涵蓋審核節點順序與最終合片閘門、接受影片路徑解析及審核訊息對延續影響的說明。`test_agentic_planner.py` 全檔 56 項通過；`test_agent_social.py`、`test_h3_prompt_delivery.py`、`test_workflow_runner.py` 共 59 項通過。`git diff --check` 通過。Ruff 未安裝，未執行 lint。

相關實作：

- [`long_video.py`](../agentic/src/agentic/runtime/planning/long_video.py)：每段接受節點及尾幀來源閘門。
- [`agent_social.py`](../agentic/src/agentic/skills/agent_social.py)：segment acceptance 審核文字、無法審核時 fail closed。
- [`agent_primitives.py`](../agentic/src/agentic/skills/agent_primitives.py)：接受審核回傳的影片路徑可供尾幀抽取。
- [`test_agentic_planner.py`](../agentic/tests/test_agentic_planner.py)：BDD 驗收案例。
- [`minimax_h3_prompting.md`](minimax_h3_prompting.md)：policy 說明及驗收語意。

## 證據邊界與下一步

本次檢查支持一項流程設計整合，不表示已驗證 Discord 即時審核、ComfyUI/H3 生成、全長成片視覺品質，或新 policy 對實際完播率的影響。若要判斷成片是否改善，下一步應在相同 brief、相同 H3 設定下比較 policy 開關兩組，保存 prompt、輸入錨點、生成片段、review 決策與 QA 結果，再由人盲評角色/場景連續性、動作可讀性和敘事結果。結構測試只回答流程契約是否成立，不能代替這項渲染實驗。
