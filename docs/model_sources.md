# 模型來源與 ComfyUI 資產

本頁記錄有公開來源 URL 的 ComfyUI 工作流模型與必要資產。新增或更換模型時，請在此列出模型頁及資產下載網址，並同步更新資產宣告：

- Krea 2 與 FLUX Klein：`configs/workflow/workflow_config.yaml` 的 `required_assets`。
- MiniMax H3 全部路線：`agentic/src/agentic/assets/minimax_h3.py`（`AssetRegistry` 依 `conditioning.h3_mode` 由此產生需求，`workflow_config.yaml` 中的 H3 項目不含 `required_assets`）。

上述檔案是安裝目錄、檔名與檔案大小的來源。`AssetRegistry` 只檢查檔案是否存在且大小相符，缺檔時回報 `manual_setup`，不會自動下載，請依下列網址手動安裝。

## Krea 2 Turbo

模型頁：[Krea 2 SVDQuant ComfyUI](https://huggingface.co/AlperKTS/Krea-2-SVDQuant-ComfyUI)

| 資產 | ComfyUI 目錄 | 下載網址 |
|---|---|---|
| `Krea2-Turbo-W4A4-noLowRank.safetensors` | `ComfyUI/models/diffusion_models` | [Hugging Face 檔案](https://huggingface.co/AlperKTS/Krea-2-SVDQuant-ComfyUI/resolve/main/checkpoints/Krea2-Turbo-W4A4-noLowRank.safetensors) |
| `Qwen3VL-4B-Instruct-Q4_K_M.gguf` | `ComfyUI/models/clip` | [Hugging Face 檔案](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen3VL-4B-Instruct-Q4_K_M.gguf) |
| `qwen_image_vae.safetensors` | `ComfyUI/models/vae` | [Hugging Face 檔案](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors) |

這些資產供 `krea2_turbo.json` 使用，設定檔另記錄 SHA-256 供核對。

## FLUX 3 Image Klein

模型頁：[FLUX 3 Image Klein ComfyUI](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy)

| 資產 | ComfyUI 目錄 | 下載網址 |
|---|---|---|
| `flux3-image-klein-int8-convrot.safetensors` | `ComfyUI/models/diffusion_models` | [Hugging Face 檔案](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/main/flux3-image-klein-int8-convrot.safetensors) |
| `qwen3vl_4b_int8_convrot.safetensors` | `ComfyUI/models/text_encoders` | [Hugging Face 檔案](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/main/qwen3vl_4b_int8_convrot.safetensors) |
| `flux2-vae.safetensors` | `ComfyUI/models/vae` | [Hugging Face 檔案](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/main/flux2-vae.safetensors) |
| `qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors` | `ComfyUI/models/text_encoders` | [Hugging Face 檔案](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/main/qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors) |

以上資產供 `flux3_image_klein_img2img.json` 使用；工作流包含 prompt upsampler，並需要 CLIPLoader 支援 `flux3` 類型的 ComfyUI（已在 0.41.0 測試）。該 Hugging Face 儲存庫需要符合其存取條件，使用與商業輸出須遵守 FLUX Kommunity License v1；詳見 [`workflow_config.yaml`](../configs/workflow/workflow_config.yaml)。FLUX Klein 產生靜態圖片，影片仍由 H3 工作流生成。

## MiniMax FastH3（T2VA、I2VA、FL2VA、L2VA）

模型頁：[FastVideo-FastH3-Comfy](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy)

| 資產 | ComfyUI 目錄 | 下載網址 |
|---|---|---|
| `fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors` | `ComfyUI/models/diffusion_models` | [Hugging Face 檔案](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy/resolve/main/diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors) |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `ComfyUI/models/text_encoders` | [Hugging Face 檔案](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) |
| `minimax_h3_video_vae_fp16.safetensors` | `ComfyUI/models/vae` | [Hugging Face 檔案](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors) |
| `minimax_h3_audio_vae_fp32.safetensors` | `ComfyUI/models/vae` | [Hugging Face 檔案](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) |

這些資產供 `minimax_h3_t2va.json`、`minimax_h3_i2va.json`、`minimax_h3_fl2va.json` 與 `minimax_h3_l2va.json` 使用。FastH3 是 8 步蒸餾模型，只能在 8 步下執行；兩個 VAE 同時供 Ref2VA 使用。

權重受 MiniMax H3 Community License 約束（部分地區不得使用），商業使用前請先閱讀其條款。

## 尚未列出公開來源的資產

Ref2VA（`minimax_h3_ref2va.json`）仍使用 Q4 GGUF 流程，下列檔名與大小由 `agentic/src/agentic/assets/minimax_h3.py` 宣告，該清單沒有公開下載網址：

| 資產 | ComfyUI 目錄 | 備註 |
|---|---|---|
| `MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf` | `ComfyUI/models/unet` | 擴散模型 |
| `qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf` | `ComfyUI/models/clip` | 文字編碼器，預設 `q4` profile |
| `qwen3vl-32B-MiniMax-H3-Q2_K.gguf` | `ComfyUI/models/clip` | 文字編碼器，`q2` profile |

取得並確認可公開引用的上游頁面後，再補入本頁；不要自行推測或拼造模型 URL。

下列工作流目前沒有在 `workflow_config.yaml` 或程式中宣告 `required_assets`，`AssetRegistry` 不會檢查其模型，本頁也沒有來源網址：

| 工作流 | 引用的模型檔 |
|---|---|
| `anima_anime.json`、`kirby_keyframe_anima.json` | `anima/anima-aesthetic-v1.1.safetensors`、`anima/qwen_3_06b_base.safetensors`、`anima/qwen_image_vae.safetensors`（`anima_anime.json` 另用 `4x-AnimeSharp.pth`） |
| `z_image_plus_nova_model.json` | `z_image/z_image_turbo_bf16.safetensors`、`qwen/qwen_3_4b.safetensors`、`flux/ae.safetensors`、`sdxl/waiIllustriousSDXL_v150.safetensors` |
| `nova_model_plus_z_image_anime.json` | `z_image/zImageTurboAnime_aioBF16.safetensors`、`sdxl/waiIllustriousSDXL_v150.safetensors`、`depth_anything_v2_vitl.pth` |
| `nova-anime-xl.json` | `sdxl/novaAnimeXL_ilV140.safetensors`、`sdxl/novaAnimeXL_ilV180.safetensors`、`sdxl/waiIllustriousSDXL_v150.safetensors`、`sdxl/reiXL_NB11.safetensors` |
| `Tile Upscaler SDXL.json` | `sdxl/noobaiXLNAIXL_vPred10Version.safetensors`、`sdxl/xinsir_controlnet-tile-sdxl-1.0.safetensors`、`sdxl/sdxl_lightning_8step_lora.safetensors` |
