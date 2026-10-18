# 模型來源與 ComfyUI 資產

本頁記錄有公開來源 URL 的 ComfyUI 圖像工作流模型與必要資產。新增或更換模型時，請在此列出模型頁及資產下載網址，並同步更新 `configs/workflow/workflow_config.yaml` 的 `required_assets`；該設定檔是安裝目錄、檔名與資產規格的來源。

## Krea 2 Turbo

模型頁：[Krea 2 SVDQuant ComfyUI](https://huggingface.co/AlperKTS/Krea-2-SVDQuant-ComfyUI)

| 資產 | ComfyUI 目錄 | 下載網址 |
|---|---|---|
| `Krea2-Turbo-W4A4-noLowRank.safetensors` | `ComfyUI/models/diffusion_models` | [Hugging Face 檔案](https://huggingface.co/AlperKTS/Krea-2-SVDQuant-ComfyUI/resolve/main/checkpoints/Krea2-Turbo-W4A4-noLowRank.safetensors) |
| `Qwen3VL-4B-Instruct-Q4_K_M.gguf` | `ComfyUI/models/clip` | [Hugging Face 檔案](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen3VL-4B-Instruct-Q4_K_M.gguf) |
| `qwen_image_vae.safetensors` | `ComfyUI/models/vae` | [Hugging Face 檔案](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors) |

這些資產供 `krea2_turbo.json` 和 `krea2_turbo_img2img.json` 共用。

## FLUX 3 Image Klein

模型頁：[FLUX 3 Image Klein ComfyUI](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy)

| 資產 | ComfyUI 目錄 | 下載網址 |
|---|---|---|
| `flux3-image-klein-int8-convrot.safetensors` | `ComfyUI/models/diffusion_models` | [Hugging Face 檔案](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/main/flux3-image-klein-int8-convrot.safetensors) |
| `qwen3vl_4b_int8_convrot.safetensors` | `ComfyUI/models/text_encoders` | [Hugging Face 檔案](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/main/qwen3vl_4b_int8_convrot.safetensors) |
| `flux2-vae.safetensors` | `ComfyUI/models/vae` | [Hugging Face 檔案](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/main/flux2-vae.safetensors) |
| `qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors` | `ComfyUI/models/text_encoders` | [Hugging Face 檔案](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/main/qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors) |

以上資產供 `flux3_image_klein_img2img.json` 使用；工作流包含 prompt upsampler。該 Hugging Face 儲存庫需要符合其存取條件，使用與商業輸出須遵守 FLUX Kommunity License v1；詳見 [`workflow_config.yaml`](../configs/workflow/workflow_config.yaml)。FLUX Klein 產生靜態圖片，影片仍由 H3 工作流生成。

## 尚未列出公開來源的資產

MiniMax H3 的必要檔名與大小目前由 `agentic/src/agentic/assets/minimax_h3.py` 宣告，該清單沒有公開下載網址。取得並確認可公開引用的上游頁面後，再補入本頁；不要自行推測或拼造模型 URL。
