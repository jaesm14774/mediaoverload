# SparkVSR × MediaOverload

研究日期：2026-09-19

## Cleanup status

本文件是本次研究與測試的唯一保留 artifact。測試期間新增的 anchor policy、planner / CLI wiring、比較腳本與 GPU 輸出，因三組 20 秒 A/B 沒有呈現穩定的視覺優勢，且 `sparse_keyframes` 的 memory retry 較多，已依決策全部還原或刪除；以下內容保留作為歷史測試紀錄，不代表目前 production code 仍包含這些改動。

## 結論

可以取其精華，但目前不應把 SparkVSR 的完整模型直接併入 MediaOverload 的 production route。

最適合移植的是它的稀疏 keyframe 思路：在整段時間軸上固定分布少量強 conditioning checkpoint，其餘時間依靠連續 latent / frame state 傳遞。MediaOverload 已經有 `rendered_tail`：每個 H3 segment 先抽出實際最後一幀，再交給下一段作 first anchor。因此本次整合只新增一個明確的 `sparse_keyframes` anchor policy，沒有改掉預設的 `transition_points`，也沒有假裝它已經是 latent video super-resolution。

## 上游研究證據

- [SparkVSR repository](https://github.com/taco-group/SparkVSR)：pipeline 是 LR video latent 加上稀疏 HR keyframe latent，再做 one-step denoise；keyframe 可來自手動選擇、codec I-frame 或 random sampling。
- [SparkVSR paper](https://arxiv.org/html/2603.16864v1)：reference guidance 越強，感知品質提高，但 PSNR / SSIM 可能下降，代表它是 perception-versus-distortion trade-off，不是無條件提升。
- [SparkVSR issue #7](https://github.com/taco-group/SparkVSR/issues/7)：官方釐清 Stage 1 是中間 latent，Stage 2 才是最後輸出；`no_ref` 是 baseline，reference modes 才是建議路徑。
- [SparkVSR weights](https://huggingface.co/JiongzeYu/SparkVSR)：repo 權重約 42.2GB。
- [CogVideoX1.5-5B-I2V base](https://huggingface.co/zai-org/CogVideoX1.5-5B-I2V)：其基礎 I2V 權重約 31.1GB。

論文的 MovieLQ 結果也支持「稀疏 reference 有效、但需控制」這個方向：SparkVSR no-reference 的 MUSIQ / DOVER 是 56.34 / 0.512；PiSA reference 是 68.88 / 0.6212。reference 數量從 0 增加到 4 時，MUSIQ 從 56.34 升到 65.76，並非需要每一幀都加 reference。

這些數字是 SparkVSR 官方資料集與 pipeline 的結果，不是 MediaOverload 生成的 anime clip 結果；本 repo 不能直接把它們當成自己的品質承諾。

## MediaOverload 實作

新增項目：

- `transition_points`：現有預設 A 組，保留原本集中在中後段的 deliberate FL2V checkpoints。
- `sparse_keyframes`：B 組，以 deterministic evenly spaced checkpoints 分布在 opening / middle / payoff；checkpoint 數由 `longvideo_sparse_keyframe_budget` 控制，預設 3。
- `ConditioningPlan`、render inputs、publish summary 都會記錄 policy、checkpoint indices 與 budget。
- CLI 可使用 `--longvideo-anchor-policy` 和 `--longvideo-sparse-keyframe-budget`。
- config 預設明確寫成 `anchor_policy: transition_points`，所以不會偷偷改變既有 route。

## A/B 結果：planner contract

執行命令：

```powershell
python scripts/compare_longvideo_anchor_policies.py --segment-count 6 --sparse-keyframe-budget 3
```

| 指標 | A: transition_points | B: sparse_keyframes |
|---|---:|---:|
| anchor indices | `[2, 4, 5]` | `[0, 3, 5]` |
| recipe sequence | `I2V, I2V, FL2V, I2V, FL2V, FL2V` | `FL2V, I2V, I2V, FL2V, I2V, FL2V` |
| deliberate anchors | 3 | 3 |
| rendered tails | 6 | 6 |
| continuation edges | 5 | 5 |
| planner nodes | 41 | 41 |

判讀：B 把一個強 anchor 從第 3 段移到 opening，並把另一個中後段 anchor 移到第 4 段；不增加目前計畫節點數，也不移除任何實際 rendered-tail handoff。這是符合 SparkVSR 精神的最小可驗證改動。

## GPU smoke evidence

先前的 2.8 秒 smoke 只用來確認 wiring：4 段、8 steps、512×288、24fps，A 組 workflow 與 final QA 通過，不能拿來判斷 15–30 秒影片的觀看品質。其歷史輸出已在 cleanup 時刪除，不再作為本次 A/B 的主要證據。

## 20 秒 paired E2E A/B

為了符合「至少 15–30 秒才看得出來」的評估條件，重新用同一個 4-segment 目標跑 A/B：

```powershell
$env:AGENTIC_LLM_MODE = 'template'
python scripts/run_longvideo_production_e2e.py --segments 4 --length 120 --steps 16 --seed 55123 --anchor-policy transition_points --sparse-keyframe-budget 3 --output-root output/sparkvsr_anchor_ab_20s_template/A --comfy-root D:\ComfyUI_windows_portable --comfy-host 127.0.0.1 --comfy-port 8188
python scripts/run_longvideo_production_e2e.py --segments 4 --length 120 --steps 16 --seed 55123 --anchor-policy sparse_keyframes --sparse-keyframe-budget 3 --output-root output/sparkvsr_anchor_ab_20s_template/B --comfy-root D:\ComfyUI_windows_portable --comfy-host 127.0.0.1 --comfy-port 8188
```

控制條件是同一個 seed `55123`、同一套 ComfyUI/H3 workflow、同一解析度與 steps；`template` mode 是為了排除外部 provider 等待，讓這一輪只比較 anchor schedule，不把 LLM story 生成品質混進來。

| 指標 | A: transition_points | B: sparse_keyframes |
|---|---:|---:|
| anchor indices | `[1, 2, 3]` | `[0, 2, 3]` |
| recipe sequence | `anchor_first, anchor_first_last, anchor_first_last, anchor_first_last` | `anchor_first_last, anchor_first, anchor_first_last, anchor_first_last` |
| segment QA | 4/4 passed | 4/4 passed |
| rendered-tail contract | passed | passed |
| final duration | 20.0 s | 20.0 s |
| final video QA | passed | passed |
| memory retry count | 1 | 2 |

報告與影片輸出已在 cleanup 時刪除；本文件保留測試條件、QA 結果、重試次數與人工觀看結論。

### 觀看檢查

實際抽看兩支 20 秒影片的 0、5、10、15、19 秒，以及每個 5 秒 segment 邊界前後的影格：

- A、B 都維持 Kirby identity；目前沒有看到明顯的硬切、黑幀、影音錯位或跨段角色崩壞。
- A 的畫面較常把道具放進中景，例如推車、球與場景物件；B 的 opening 與中段較常使用 Kirby close-up。這是單次 paired render 的觀察，不足以判定哪個 policy 的感知品質較高。
- B 沒有在 20 秒觀看長度上呈現「明顯勝過 A」的證據；它已從 planner-only 變成可實際 render、可觀看、可 QA 的候選 route。

因此單組結果的結論是：`sparse_keyframes` 已證明 production wiring 與 20 秒 E2E 可行，但不宣稱視覺品質勝出。

## 三組 paired E2E follow-up

依照 15–30 秒影片才適合觀察的原則，再補跑兩組 seed；加上前一組，總共比較三組 A/B。六支影片都使用同一份 goal prompt、20 秒目標、512×288、24fps、16 steps、template mode，只有 base seed 與 anchor policy 改變。

| seed | A final QA | B final QA | A memory retries | B memory retries | 視覺抽查 |
|---:|---|---|---:|---:|---|
| 55123 | passed | passed | 1 | 2 | A/B 都維持 identity，沒有穩定勝者 |
| 66132 | passed | passed | 0 | 2 | B 的構圖較穩定，仍不足以判定品質勝出 |
| 77141 | passed | passed | 0 | 2 | A/B 都連貫，但故事物件與動作偏好不同 |

Aggregate：A 是 3/3 run、12/12 segment QA、總 memory retry 1 次；B 是 3/3 run、12/12 segment QA、總 memory retry 6 次。兩邊 final video 都是 20.0 秒，rendered-tail contract 全部通過。

六張 contact sheet 的人工抽查沒有發現 B 在三個 seed 都穩定改善 identity 或接縫；反而 B 在這台 8GB GPU 上較常觸發 memory retry。這表示 B 是可工作的候選 route，但目前沒有足夠證據取代 A。

這仍不是嚴格的 pixel-matched perceptual benchmark：anchor policy 會改變 I2V/FL2V 的工作流順序，因此 A/B 可能生成不同道具與動作。這三組更適合解讀為「route robustness + 實際可觀看性」比較，而非固定內容的 PSNR/SSIM 實驗。

## 尚未宣稱的結果

本機目前沒有 SparkVSR ComfyUI node / model assets；而且完整 SparkVSR 加 CogVideoX I2V 的權重規模不適合在目前 8GB RTX 4060 上直接當 production dependency。因此這次整合的是「稀疏 reference / keyframe schedule」的精華，不是把 SparkVSR 模型本體併入。

下一個合理實驗是固定 storyboard 與 anchor image，再用多個 seed 與相同 review rubric 重跑 A/B，量 identity consistency、尾幀接縫、主體保持率、生成時間與 VRAM peak；目前維持 `transition_points` 為預設，將 `sparse_keyframes` 作為可明確選擇的實驗 route。
