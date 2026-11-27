# 文字 LLM 供應商與備援鏈

文字請求先送給主模型，失敗再依序換下一個。所有供應商共用同一個 OpenAI Chat Completions 轉接器（`agentic/src/agentic/runtime/model_backends.py` 的 `OpenAICompatibleModel`），新增供應商只要在 `OPENAI_COMPATIBLE_PROVIDER_SPECS` 加一筆，不需要新類別。

## 供應商

| provider | 金鑰環境變數 | 預設文字模型 | 說明 |
|---|---|---|---|
| `experiential` | `EXPERIENTIAL_API_KEY` | `gpt-6-luna` | 閘道 `https://api.experientiallabs.ai/v1`。免費方案每天 1M token，GPT-6 Luna、DeepSeek V4.1 Flash、Clef 共用同一個池，UTC 00:00 重置，每分鐘 10 次請求、250K token；免費模型需先做信用卡驗證 |
| `groq` | `GROQ_API_KEY` | `qwen/qwen3.8-27b` | 不支援 `json_schema` |
| `gemini` | `GEMINI_API_KEY` | `gemini-3.5-flash-lite` | 支援視覺 |
| `cloudflare` | `CLOUDFLARE_API_TOKEN`、`CLOUDFLARE_ACCOUNT_ID` | `@cf/openai/gpt-oss-120b` | 預設輸出上限只有 256 token，推理模型會全部花在推理，所以規格內固定 `max_tokens` 為 16384 |
| `nvidia`、`amd`、`openai` | 見 `model_backends.py` | 見 `model_backends.py` | |

`experiential` 只當文字供應商：不設 `max_tokens`（Luna 的推理 token 算在輸出上限內，上限太小會得到空內容），視覺與 `json_schema` 都關閉，直到對真實閘道驗證過。閘道目前不支援調整推理強度。

## 鏈的組成

- 主模型：`AGENTIC_TEXT_MODEL_PROVIDER` 與 `AGENTIC_TEXT_MODEL`。
- 備援：`AGENTIC_TEXT_FALLBACK_PROVIDERS` 與 `AGENTIC_TEXT_FALLBACK_MODELS`，兩者以逗號分隔且**長度必須相同**，否則所有備援被丟棄，只記錄 `text_fallback_config_error`。備援最多 5 個，缺金鑰的會被略過並記入 `text_fallback_skipped`。
- 備援只在 `AGENTIC_PROVIDER_FALLBACK_ENABLED` 或 `AGENTIC_TEXT_ALLOW_FALLBACK` 為 true 時啟用。
- 視覺預設跟隨文字供應商；`experiential` 不支援視覺，所以必須明確設定 `AGENTIC_VISION_MODEL_PROVIDER`。
- 主模型缺金鑰是硬失敗，不會自動換下一個。
- 總期限（`AGENTIC_LLM_TOTAL_TIMEOUT_SECONDS`）平均分給仍可用的供應商，單一供應商不會吃光整個期限。

## token 帳

每次呼叫的 `provider_attempts`（存在該次 run 的 `llm/*.json`）都帶 `tokens`：`prompt`、`completion`、`reasoning`、`total`。同一次呼叫內的驗證重試會累加，因為每次重試都會再花配額；答案無法使用（例如推理用盡輸出上限、內容為空）時仍會記錄，因為那些 token 已經從配額扣掉。供應商沒回報用量時不會出現 `tokens`。`reasoning` 包含在 `completion` 裡。

加總整個 run 的 token：把該 run 資料夾 `llm/*.json` 內每個 attempt 的 `tokens.total` 相加。
