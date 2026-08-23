# 🥷 MediaOverload：Windows 全新電腦 ComfyUI 重建手冊

> 文件版本：2026-08-23
>
> 適用 repo：`MediaOverload`
>
> 目的：在一台沒有既有 ComfyUI、沒有既有 model、沒有既有 custom node 的 Windows 電腦上，重建本 repo 可以使用的 ComfyUI runtime。
>
> 重要：本手冊把「目前實際可重建的基線」與「repo 還保留但來源沒有完整鎖定的舊 workflow」分開。不要因為某個 workflow JSON 仍然存在，就推論它的所有 model URL、版本與 checksum 已經被 repo 記錄。

## 0. 先看結論

### 0.1 推薦的第一個可工作的安裝範圍

先完成下面這個基線，才能在 RTX 4060 8 GB 類似的 Windows 機器上穩定驗證目前主線：

| 能力 | ComfyUI 資產 | 狀態 |
|---|---|---|
| Krea 2 Turbo 首幀/生圖 | Krea GGUF、Qwen3-VL 4B GGUF、Qwen Image VAE | 已在目前 D: runtime 實際存在並驗證 |
| MiniMax H3 低 VRAM T2V/I2V/15 秒 FL2VA | H3 Q4 diffusion、H3 Q4 text、video VAE、audio VAE | 已在目前 D: runtime 實際存在 |
| MiniMax H3 Ref2VA | Ref2VA Q4 diffusion、H3 Q4 text、兩個 VAE、VideoHelperSuite | 已在目前 D: runtime 實際存在 |
| H3 Spectrum 記憶體優化 | ComfyUI-Spectrum-MiniMax-H3 | 已在目前 D: runtime 載入 |

這個基線是目前 repo 最有證據的路徑：Krea 2 先產生可審核 still，再把核准的 frame 交給 MiniMax H3；Krea 2 不是本機 video model。repo 對這個邊界已有說明：[Krea 2 / H3 local best practice](./krea2_comfyui_best_practice.md)。

### 0.2 儲存空間與硬體預算

- 建議安裝磁碟至少預留 **100 GB**。
- 只裝上面的主線基線，model 權重大約 50 GB，還要加上 ComfyUI、custom node、Python packages、暫存下載與輸出。
- `native-quality` H3 另外需要約 39.55 GiB 的 native diffusion/text 權重；不要在 8 GB GPU 上把 native profile 當成預設。
- 目前實機證據是：RTX 4060、8 GB VRAM、ComfyUI 0.30.0、embedded Python 3.13.12、PyTorch 2.11.0+cu130。這不是「任何 Windows 電腦都保證成功」的硬體承諾；CPU、AMD GPU、沒有 CUDA 的電腦不符合本手冊的主線目標。

### 0.3 本文件的完整性聲明

截至 2026-08-23，本 repo 的 workflow inventory 還有以下重建缺口：

1. `reiXL_NB11.safetensors` 只在 `configs/workflow/nova-anime-xl.json` 和 `scripts/evaluate_kirby_roles.py` 被引用；repo 沒有下載 URL、SHA256、版本、CivitAI model/version ID；目前 D: ComfyUI 也沒有這個檔案。
2. `comfyui_controlnet_aux`、`comfyui-easy-use`、`comfyui-advanced-controlnet`、`comfyui-videohelpersuite` 等目前資料夾有些沒有保留自己的 Git commit，因此不能宣稱只有用 `git clone` 就能得到 byte-identical runtime。
3. 多個舊 SDXL/Nova workflow 只在 JSON 寫了 model filename，沒有由 repo manifest 管理 model source、大小或 checksum。

因此，本手冊對「主線基線」提供完整可執行步驟；對缺口則列出目前能確認的來源，或明確寫 `BLOCKED / URL NOT RECORDED`。安裝者不可用不明來源替換檔案後直接宣稱與目前 runtime 相同。

## 1. Source of truth：安裝時只以哪些檔案為準

先把 repo 置於一個固定位置，例如：

```powershell
$Repo = 'C:\Work\mediaoverload'
Set-Location $Repo
```

以下檔案是 ComfyUI 整合的主要來源：

| 檔案 | 內容 |
|---|---|
| `configs/routing.yaml` | workflow 權重、候選策略與哪些路由已停用 |
| `configs/workflow/*.json` | 實際 ComfyUI API graph、node class、model filename |
| `configs/workflow/workflow_config.yaml` | 目前正式 manifest；現階段主要完整管理 Krea 2 資產 |
| `agentic/src/agentic/assets/minimax_h3.py` | H3 profile、下載 URL、目標目錄、大小、部分 SHA |
| `scripts/setup_minimax_h3.py` | H3 可重試、可續傳、profile-aware 的安裝器 |
| `scripts/run_comfyui_h3_lowvram.ps1` | 本 repo 的 H3 low-VRAM 啟動參數 |
| `docs/krea2_comfyui_best_practice.md` | Krea 2 → H3 邊界、Krea graph 設定與 `/free` lifecycle |
| `media_overload.env.example` | repo app/scheduler 與 ComfyUI host 的設定範例 |
| `docker-compose.yml` | Docker scheduler 如何透過 `host.docker.internal:8188` 連宿主機 ComfyUI |

**不要**把另一台電腦的 `extra_model_paths.yaml` 直接複製過來。現在實機的檔案指向 `E:/comfyui_extra`，但該磁碟在目前環境不存在；全新安裝應先使用標準 `ComfyUI/models` 目錄。

## 2. 安裝前準備

### 2.1 Windows 與驅動

1. 安裝 64-bit Windows 10/11。
2. 安裝支援目前 NVIDIA CUDA runtime 的 NVIDIA driver，安裝後重開機。
3. 開啟 PowerShell，確認 Git、`curl.exe`、`py` 可見：

```powershell
git --version
curl.exe --version
python --version
```

4. 如果 `git` 或 `python` 不存在，先安裝 Git for Windows 與 Python 3.11+。ComfyUI portable 自帶 embedded Python；repo app 的 virtual environment 是另一個 Python，不要混用。
5. 確認磁碟有至少 100 GB 可用空間；下載中途需要額外 `.part` 暫存檔。

### 2.2 取得 repo

在全新機器上使用正式 remote/branch；不要把目前工作機的未提交修改當成安裝依賴：

```powershell
$MediaOverloadRemote = Read-Host 'MediaOverload git URL'
git clone $MediaOverloadRemote C:\Work\mediaoverload
Set-Location C:\Work\mediaoverload
git status --short --branch
```

如果本 repo 的分支尚未推送，先由團隊決定要安裝哪一個 commit；本手冊不能替未提交的 workflow/model 變更自動做版本選擇。

## 3. 安裝 ComfyUI portable：版本與目錄

### 3.1 下載與解壓

目前官方 Windows portable 文件是：[ComfyUI Portable Windows](https://docs.comfy.org/installation/comfyui_portable_windows)。官方 portable 包含 embedded Python；NVIDIA portable 的標準路徑是 CUDA 13 / Python 3.13 方向。請從官方頁面取得最新合法下載連結，不要從第三方 re-pack 取得 executable。

把 portable 解壓到固定、沒有空白與特殊字元的路徑；本手冊以 `D:\ComfyUI_windows_portable` 為例：

```powershell
$ComfyRoot = 'D:\ComfyUI_windows_portable'
$Comfy = Join-Path $ComfyRoot 'ComfyUI'
$ComfyPython = Join-Path $ComfyRoot 'python_embeded\python.exe'

Test-Path $Comfy
Test-Path $ComfyPython
```

若你使用別的磁碟，之後所有 `$ComfyRoot` 都要換成同一個絕對路徑；不要只改環境變數而遺漏 script argument。

### 3.2 鎖定目前 repo 已驗證的 ComfyUI source

目前工作機的實際 evidence：

```text
ComfyUI version: 0.30.0
ComfyUI git remote: https://github.com/Comfy-Org/ComfyUI
ComfyUI git commit: 531ea7db139a856a830182694441e9755f0e260a
embedded Python: 3.13.12
PyTorch: 2.11.0+cu130
CUDA: 13.0
GPU: NVIDIA GeForce RTX 4060
frontend: 1.48.6 required and installed
workflow templates: 0.11.31 required and installed
```

全新機器若要重建這個 exact source，portable 解壓後進行：

```powershell
Set-Location $Comfy
git fetch --all --tags
git checkout 531ea7db139a856a830182694441e9755f0e260a
```

注意：portable release 不一定是完整 Git checkout。如果 `git checkout` 失敗，不能用「看起來接近的最新版本」默默替代；請保留官方 portable 版本資訊與目前 commit，並把差異記錄在安裝 manifest。ComfyUI 官方也持續更新 Python/PyTorch/CUDA 組合，因此「最新 portable」與「目前已驗證的 0.30.0」是兩個不同選項。

### 3.3 建立標準目錄

```powershell
$ModelDirs = @(
    'checkpoints', 'clip', 'controlnet', 'diffusion_models', 'embeddings',
    'loras', 'models', 'model_patches', 'text_encoders', 'unet',
    'upscale_models', 'vae', 'vae_approx'
)

foreach ($Name in $ModelDirs) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Comfy "models\$Name") | Out-Null
}
New-Item -ItemType Directory -Force -Path (Join-Path $Comfy 'custom_nodes') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Comfy 'output\mediaoverload') | Out-Null
```

初次安裝不要建立或複製 `extra_model_paths.yaml`。只有在所有模型已整理到獨立磁碟、並且你明確知道每個 path 的用途時，才採用官方 [extra model paths](https://docs.comfy.org/installation/desktop/windows) 設定；改完必須重啟 ComfyUI。

### 3.4 驗證 embedded Python / GPU

```powershell
& $ComfyPython --version
& $ComfyPython -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.version.cuda); print('available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

預期至少要看到：`Python 3.13.x`、`torch 2.11.0+cu130` 或一個與所選 ComfyUI portable 相容的 NVIDIA build、`available=True`。如果 `available=False`，先修復 driver/PyTorch/portable，不要開始下載 50 GB model。

## 4. 安裝 custom nodes / plugins

### 4.1 安裝原則

- 所有 ComfyUI custom node package 都安裝到 `$Comfy\custom_nodes`。
- custom node 的 Python dependency 安裝到 **ComfyUI embedded Python**，不是 repo `.venv`。
- 每個 node 安裝後都要記錄 `git remote`, `git rev-parse HEAD`, `requirements.txt` 安裝結果。
- 目前 machine 有一些額外 node，但 workflow 不引用它們；本手冊不把它們列為必要依賴，避免新增不必要的 CUDA/ONNX 相容性問題。
- `comfyui-logicutils.disabled` 是停用目錄，不要複製成 active node。

### 4.2 目前 workflow 實際需要的 node

| Node | 來源 | 目前 machine evidence | 用途 |
|---|---|---|---|
| ComfyUI-GGUF | [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | `72c8990f22b86b06a4c9f4cad628d18825160f79` | `UnetLoaderGGUF`, `CLIPLoaderGGUF`；Krea/H3 Q4/Q2 |
| Spectrum H3 | [xmarre/ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3) | `66d0c7a5ffe9aeded072fd7480cf17b832221055` | H3 的 `SpectrumApplyMiniMaxH3` |
| VideoHelperSuite | [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | 目錄存在，但目前未保留 Git commit | Ref2VA 的 `VHS_LoadVideoPath`、video helper |
| ControlNet Aux | [Fannovel16/comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) | 目錄含本地修改，無可用 commit | `DepthAnythingV2Preprocessor` |
| Advanced ControlNet | [Kosinkadink/ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet) | 目錄存在，未保留可驗證 commit | `ACN_ReferenceControlNet`, `ACN_ReferencePreprocessor` |
| Easy-Use | [yolain/ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use) | version `1.3.6`，未保留 commit | `easy comfyLoader`, `easy fullkSampler`, `easy controlnetLoader` |
| Essentials | [cubiq/ComfyUI_essentials](https://github.com/cubiq/ComfyUI_essentials) | `9d9f4bedfc9f0321c19faf71855e228c93bd0dc9` | `ImageTile+`, `ImageUntile+`, `ImageListToBatch+` |
| Impact Pack | [ltdrdata/ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) | `429d0159ad429e64d2b3916e6e7be9c22d025c3c` | `ImpactImageBatchToImageList` |
| rgthree | [rgthree/rgthree-comfy](https://github.com/rgthree/rgthree-comfy) | 目錄存在，未保留 commit | `Image Comparer (rgthree)` |
| WAS | [ltdrdata/was-node-suite-comfyui](https://github.com/ltdrdata/was-node-suite-comfyui) | `44de705818d4663fefefde57ffe0ea5a9ea39df4` | `Image Lucy Sharpen` |
| ComfyUI Manager | [ltdrdata/ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) | `15ec9a901b47953f686805674bba26da65cd252b` | 維護/管理工具，不是目前 graph 的必要 class |

`ComfyUI-GGUF` 的 model URL 與 profile metadata 中曾出現的 `molbal/ComfyUI-GGUF` 不要混淆：目前 live ComfyUI 載入的是 city96 fork 的 `ComfyUI-GGUF`，以此為重建基準；H3 權重本身仍依照 `minimax_h3.py` 指定的 model repository 下載。

### 4.3 目前機器已安裝、但不是 current workflow 必要依賴的 plugins

這些目錄在目前 D: ComfyUI 存在，但逐一對照 configs/workflow/*.json 的 class_type 後，沒有被目前 repo graph 使用。它們是「現場已安裝」而不是「全新基線必裝」：

| 目前目錄 | upstream / observed revision | 判定 |
|---|---|---|
| ComfyUI_IPAdapter_plus | cubiq/ComfyUI_IPAdapter_plus；commit a0f451a5113cf9becb0847b92884cb10cbdec0ef | current workflow 沒有 IPAdapter class；不必安裝 |
| ComfyUI-WanVideoWrapper | kijai/ComfyUI-WanVideoWrapper；commit df8f3e49daaad117cf3090cc916c83f3d001494c | repo 已移除 WAN2.2 路徑；不必安裝 |
| comfyui-kjnodes | kijai/ComfyUI-KJNodes；現場未保留 commit | current workflow inventory 沒有 KJ class；不必安裝 |
| comfyui-multigpu | pollockjj/ComfyUI-MultiGPU；現場未保留 commit | 不是單卡 RTX 4060 基線依賴；不必安裝 |
| ComfyUI-nunchaku | nunchaku-tech/ComfyUI-nunchaku；現場未保留 commit | current graph 沒有 Nunchaku class；不必安裝 |
| comfyui-ultimatesdupscale | ssitu/ComfyUI_UltimateSDUpscale；現場未保留 commit | current Tile graph 沒有 UltimateSDUpscale class；不必安裝 |
| flashvsr_ultra_fast | lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast；現場未保留 commit | current graph 沒有 FlashVSR class；不必安裝 |
| comfyui-logicutils.disabled | 目錄名稱本身表示 disabled | 不要啟用或改名 |

ControlNet Aux 目錄內目前還有 body_pose_model.pth、facenet.pth、hand_pose_model.pth 等 annotator assets；current repo workflow 只明確需要 Depth Anything V2 的 depth_anything_v2_vitl.pth。不要把「plugin 下載了額外 annotator」誤判成 repo 每條路由都需要它們。

### 4.4 Clone node repositories

先定義 helper。這段只會建立不存在的目錄；如果目錄已存在，先停下來檢查，不要覆蓋既有 node：

```powershell
$NodeRoot = Join-Path $Comfy 'custom_nodes'

function Clone-ComfyNode {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string]$Url,
        [string]$Commit
    )
    $Target = Join-Path $NodeRoot $Name
    if (Test-Path $Target) {
        throw "Target already exists; inspect manually before continuing: $Target"
    }
    git clone $Url $Target
    if ($Commit) {
        git -C $Target fetch --all --tags
        git -C $Target checkout $Commit
    }
}

Clone-ComfyNode 'ComfyUI-GGUF' 'https://github.com/city96/ComfyUI-GGUF.git' '72c8990f22b86b06a4c9f4cad628d18825160f79'
Clone-ComfyNode 'ComfyUI-Spectrum-MiniMax-H3' 'https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git' '66d0c7a5ffe9aeded072fd7480cf17b832221055'
Clone-ComfyNode 'ComfyUI_essentials' 'https://github.com/cubiq/ComfyUI_essentials.git' '9d9f4bedfc9f0321c19faf71855e228c93bd0dc9'
Clone-ComfyNode 'ComfyUI-Impact-Pack' 'https://github.com/ltdrdata/ComfyUI-Impact-Pack.git' '429d0159ad429e64d2b3916e6e7be9c22d025c3c'
Clone-ComfyNode 'comfyui-manager' 'https://github.com/ltdrdata/ComfyUI-Manager.git' '15ec9a901b47953f686805674bba26da65cd252b'
Clone-ComfyNode 'was-node-suite-comfyui' 'https://github.com/ltdrdata/was-node-suite-comfyui.git' '44de705818d4663fefefde57ffe0ea5a9ea39df4'

# These directories did not retain a verifiable commit in the current machine.
# Clone them, then record the resulting commit before proceeding.
Clone-ComfyNode 'comfyui-videohelpersuite' 'https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git'
Clone-ComfyNode 'comfyui_controlnet_aux' 'https://github.com/Fannovel16/comfyui_controlnet_aux.git'
Clone-ComfyNode 'comfyui-advanced-controlnet' 'https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git'
Clone-ComfyNode 'comfyui-easy-use' 'https://github.com/yolain/ComfyUI-Easy-Use.git'
Clone-ComfyNode 'rgthree-comfy' 'https://github.com/rgthree/rgthree-comfy.git'

Get-ChildItem $NodeRoot -Directory | ForEach-Object {
    $GitDir = Join-Path $_.FullName '.git'
    if (Test-Path $GitDir) {
        [pscustomobject]@{
            Name = $_.Name
            Remote = (git -C $_.FullName remote get-url origin 2>$null)
            Commit = (git -C $_.FullName rev-parse HEAD 2>$null)
        }
    }
}
```

如果 upstream 的實際資料夾名稱不同，使用 `git clone <url> <expected-folder>` 明確指定資料夾名稱，讓 log 與 workflow 依賴容易檢查。不要把 Manager 產生的暫存 `.disabled`、`custom_nodes.bak` 或個人 plugin 複製進新機。

### 4.5 安裝 custom node requirements

先安裝 ComfyUI-GGUF 與所有上述 node 的 requirements。這些安裝會使用 `$ComfyPython`：

```powershell
& $ComfyPython -m pip install --upgrade pip

$RequiredNodeNames = @(
    'ComfyUI-GGUF',
    'ComfyUI-Spectrum-MiniMax-H3',
    'ComfyUI_essentials',
    'ComfyUI-Impact-Pack',
    'comfyui-manager',
    'was-node-suite-comfyui',
    'comfyui-videohelpersuite',
    'comfyui_controlnet_aux',
    'comfyui-advanced-controlnet',
    'comfyui-easy-use',
    'rgthree-comfy'
)

foreach ($Name in $RequiredNodeNames) {
    $Requirements = Join-Path $NodeRoot "$Name\requirements.txt"
    if (Test-Path $Requirements) {
        & $ComfyPython -m pip install -r $Requirements
        if ($LASTEXITCODE -ne 0) { throw "requirements failed: $Name" }
    }
}

& $ComfyPython -m pip check
```

已知 requirements 可能拉入的套件包括 `gguf`, `sentencepiece`, `protobuf`, `opencv`, `scipy`, `onnxruntime-gpu`, `diffusers`, `accelerate`, `segment-anything`, `scikit-image`, `transformers`、WAS 的影像套件與 Impact Pack 的 SAM2 dependency。這些不是 repo app 的 `requirements.txt`；不要把它們誤裝進 repo `.venv`，也不要為了「看起來完整」安裝目前 workflow 不用的 WanVideo/Nunchaku/FlashVSR 套件。

### 4.6 啟動一次，確認 node class

第一次可用官方 `run_nvidia_gpu.bat` 做最小載入測試；主線 H3 應改用第 7 節 repo launcher。啟動後執行：

```powershell
$Health = Invoke-RestMethod 'http://127.0.0.1:8188/system_stats'
$Info = Invoke-RestMethod 'http://127.0.0.1:8188/object_info'

$Health.devices
$Health.system.comfyui_version

@('UnetLoaderGGUF','CLIPLoaderGGUF','MiniMaxH3ImageToVideo',
  'MiniMaxH3ReferenceToVideo','MiniMaxH3SigmaShift','SpectrumApplyMiniMaxH3',
  'SaveVideo','VHS_LoadVideoPath','DepthAnythingV2Preprocessor') |
  ForEach-Object { "$_ : $($null -ne $Info.$_)" }
```

如果某個 class 是 `False`，先修 custom node 載入錯誤再下載 model；缺 node 與缺 model 是兩種不同問題。

### 4.7 Workflow 使用的 ComfyUI core nodes

以下 class_type 由 ComfyUI core 提供，不要另外從第三方 custom node 安裝同名替代品：

BasicGuider、BasicScheduler、CheckpointLoaderSimple、CLIPTextEncode、ConditioningZeroOut、ControlNetApplyAdvanced、CreateVideo、EmptyLatentImage、EmptySD3LatentImage、ImageScaleBy、ImageUpscaleWithModel、KSamplerAdvanced、KSamplerSelect、LoadImage、LoraLoader、ModelPatchLoader、ModelSamplingAuraFlow、PreviewImage、PrimitiveInt、PrimitiveString、QwenImageDiffsynthControlnet、RandomNoise、SamplerCustomAdvanced、SaveImage、SaveVideo、UpscaleModelLoader、VAEDecode、VAEDecodeAudio、VAEEncode、VAELoader。

CreateVideo、SaveVideo、H3 core classes 的可用性仍然跟 ComfyUI source version 有關；core node 不代表任意舊版 ComfyUI 都能執行目前 JSON。

## 5. Model inventory：檔名、目錄、URL、大小、SHA 與現況

### 5.1 主線基線：Krea 2

`workflow_config.yaml` 已提供完整 URL、預期 byte 與 SHA256。目標路徑不可更名，因為 JSON loader 使用以下 filename：

| 目標相對於 `ComfyUI\models` | URL | Expected bytes | SHA256 | 目前實機 |
|---|---|---:|---|---|
| `unet\krea2_turbo_bf16-Q4_0.gguf` | [HF molbal/krea2-gguf](https://huggingface.co/molbal/krea2-gguf/resolve/main/krea2_turbo_bf16-Q4_0.gguf) | 8,314,016,800 | `3d28960e1f3385b27a224c6968a0acc02b07194758247e576ee0c5caaab4f1cf` | FOUND |
| `clip\Qwen3VL-4B-Instruct-Q4_K_M.gguf` | [HF Qwen/Qwen3-VL-4B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen3VL-4B-Instruct-Q4_K_M.gguf) | 2,497,281,664 | `66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a` | FOUND |
| `vae\qwen_image_vae.safetensors` | [HF Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors) | 253,806,246 | `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f` | FOUND |

Krea graph 的目前設定：T2I 1024×576、8 steps、Euler、simple scheduler、CFG 1、denoise 1；img2img 使用同一組 model，denoise 0.25。先不要直接提高到 1344×768；先通過 1024×576 smoke test。

### 5.2 主線基線：MiniMax H3 low-VRAM / Ref2VA

以下 URL 與預期大小來自 `agentic/src/agentic/assets/minimax_h3.py`。其中 Comfy-Org 是 official source；`molbal`、`realrebelai`、`Abiray` 是目前 profile 指向的 community source，請保留來源名稱，不要換成另一個同名 mirror。

| 目標相對於 `ComfyUI\models` | URL | Expected bytes | Source / profile |
|---|---|---:|---|
| `vae\minimax_h3_video_vae_fp16.safetensors` | [Comfy-Org/MiniMax-H3 video VAE](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors) | 5,207,808,496 | official；所有 H3 profiles |
| `vae\minimax_h3_audio_vae_fp32.safetensors` | [Comfy-Org/MiniMax-H3 audio VAE](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) | 605,254,808 | official；所有 H3 profiles |
| `unet\minimax_h3_fl2va_pruned_fp8_Q4_0.gguf` | [molbal/MiniMax-H3-GGUF](https://huggingface.co/molbal/MiniMax-H3-GGUF/resolve/main/minimax_h3_fl2va_pruned_fp8_Q4_0.gguf) | 11,377,542,880 | community；`balanced-lowvram`, `ultra-lowvram` |
| `clip\qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf` | [realrebelai/MiniMax-H3_GGUFs Q4](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs/resolve/main/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf) | 14,576,977,888 | community；Q4 profiles |
| `clip\qwen3vl-32B-MiniMax-H3-Q2_K.gguf` | [realrebelai/MiniMax-H3_GGUFs Q2](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs/resolve/main/qwen3vl-32B-MiniMax-H3-Q2_K.gguf) | 8,487,968,160 | community；`ultra-lowvram` / `model_profile: q2` |
| `unet\MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf` | [Abiray/MiniMax-H3-Pruned-GGUF Ref2VA](https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF/resolve/main/MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf) | 11,564,180,576 | community；`ref2va-lowvram` |

H3 profile 的實際設定：

| Profile | Diffusion/text | 解析度 | length / fps | steps | 估計 model size |
|---|---|---:|---:|---:|---:|
| `balanced-lowvram` | Q4 diffusion + Q4 text + 2 VAE | 608×352 | 124 / 24 | 20 | 29.59 GiB |
| `ultra-lowvram` | Q4 diffusion + Q2 text + 2 VAE | 608×352 | 124 / 24 | 20 | 23.92 GiB |
| `ref2va-lowvram` | Ref2VA Q4 + Q4 text + 2 VAE | 608×352 | 124 / 24 | 20 | 29.76 GiB |
| `native-quality` | native diffusion + native text + 2 VAE | 608×352 | 124 / 24 | 20 | 39.55 GiB，另行下載 |

15 秒 FL2VA graph 使用 length 362、fps 24、steps 16；這不是把一般 124-frame graph 的 length 隨意改大即可取代的設定。

### 5.3 Native H3：可重建但不是目前 8 GB 基線

目前 `$Comfy\models\diffusion_models` 與 `$Comfy\models\text_encoders` 是空的，因此 native route 在目前機器 **未 ready**。只有在確定 GPU、RAM、磁碟與顯存策略允許後才下載：

| 目標 | URL | Expected bytes |
|---|---|---:|
| `diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors` | [Comfy-Org/MiniMax-H3 native diffusion](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) | 20,970,379,616 |
| `text_encoders\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | [Comfy-Org/MiniMax-H3 native text encoder](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) | 15,687,142,551 |
| `diffusion_models\minimax_h3_ref2va_pruned_int8_convrot.safetensors` | [Comfy-Org/MiniMax-H3 native Ref2VA](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) | source metadata 未提供固定 byte |

Native graph 使用 core `UNETLoader` / `CLIPLoader`，不是 GGUF 的 `UnetLoaderGGUF` / `CLIPLoaderGGUF`；兩套 loader 不可交叉替換。

### 5.4 Anima workflow

目前這些檔案在 D: runtime 缺失，但 workflow 仍保留：

| Workflow filename | Target path | Source |
|---|---|---|
| `anima-aesthetic-v1.1.safetensors` | `models\diffusion_models\anima\` | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-aesthetic-v1.1.safetensors) |
| `qwen_3_06b_base.safetensors` | `models\text_encoders\anima\` | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors) |
| `qwen_image_vae.safetensors` | `models\vae\anima\` | [circlestone-labs/Anima VAE](https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors)；也可複製已下載的 Krea VAE，但需保留 `anima\` 子目錄 |
| `4x-AnimeSharp.pth` | `models\upscale_models\` | [ComfyUI Manager model list entry](https://github.com/Comfy-Org/ComfyUI-Manager/blob/main/model-list.json)；direct URL 為 [Kim2091/AnimeSharp](https://huggingface.co/Kim2091/AnimeSharp/resolve/main/4x-AnimeSharp.pth) |

Anima graph 還固定了 `cfg=3.5`；Kirby identity refine 使用 denoise 0.35，continuity 使用 denoise 0.18；這些是 workflow graph 設定，不是 model download option。

### 5.5 SDXL / Nova / Z-Image / Tile workflow

這些是 repo JSON 中仍被引用的每一個 filename。只有有明確可追溯 source 的項目才列 direct URL；不明來源不可當成已驗證主線。

補充：nova_model_plus_z_image_anime.json 另引用 zImageTurboAnime_aioBF16.safetensors，目標是 models\checkpoints\z_image\；目前可追溯的 community mirror 是 [torestinbar/z-image-turbo](https://huggingface.co/torestinbar/z-image-turbo/resolve/main/checkpoints/zImageTurboAnime_aioBF16.safetensors)，SHA256 eae8fe30a8b2e3b6258b921fcb1d612cc737c6ce3634925d762a0aa376128a36。repo 本身未記錄 canonical source。

補充：Tile Upscaler SDXL 另引用 noobaiXLNAIXL_vPred10Version.safetensors，目標是 models\checkpoints\sdxl\；目前可追溯的 community mirror 是 [Toc/toc](https://huggingface.co/Toc/toc/resolve/main/models/noobaiXLNAIXL_vPred10Version.safetensors)，SHA256 ea349eeae87ca8d25ba902c93810f7ca83e5c82f920edf12f273af004ae02819。repo 本身未記錄 canonical source。

z_image_plus_nova_model.json 的三個 Z-Image split files 必須保留 workflow 的子目錄名稱：qwen_3_4b.safetensors 放在 models\text_encoders\qwen\，ae.safetensors 放在 models\vae\flux\，z_image_turbo_bf16.safetensors 放在 models\diffusion_models\z_image\。ComfyUI 的 folder_paths 同時搜尋 text_encoders/clip 與 diffusion_models/unet，但 JSON 中的 qwen、flux、z_image 前綴不可刪掉。

| Filename | Target path | URL / status |
|---|---|---|
| `waiIllustriousSDXL_v150.safetensors` | `models\checkpoints\sdxl\` | [HF mirror](https://huggingface.co/frankjoshua/waiIllustriousSDXL_v150/resolve/main/waiIllustriousSDXL_v150.safetensors)，SHA256 `befc694a296f75e996488ebf9f9db8a1493bd059b6e704b975829e87d5aeb4fa`；repo 本身未記錄 canonical source |
| `novaAnimeXL_ilV180.safetensors` | `models\checkpoints\sdxl\` | [HF mirror](https://huggingface.co/frankjoshua/novaAnimeXL_ilV180/resolve/main/novaAnimeXL_ilV180.safetensors)，SHA256 `a6b545d4776ccf1617091170be643e2c9a43877f9687c9224d8116c80f817d4d` |
| `novaAnimeXL_ilV140.safetensors` | `models\checkpoints\sdxl\` | [HF mirror](https://huggingface.co/frankjoshua/novaAnimeXL_ilV140/resolve/main/novaAnimeXL_ilV140.safetensors)；SHA256 `aaa94191eb317c68193c19a28b5b81c2165dbcc5e8775dde8823cc7a90b3d524` |
| `reiXL_NB11.safetensors` | `models\loras\sdxl\` | **BLOCKED**：repo、Git 歷史與目前 runtime 都沒有 URL、model ID 或 checksum；不要猜測下載。 |
| `noobaiXLNAIXL_vPred10Version.safetensors` | `models\checkpoints\sdxl\` | repo 未記錄 canonical source；目前沒有 installed file。先取得原始 model page 與 SHA 再安裝。 |
| `sdxl_lightning_8step_lora.safetensors` | `models\loras\sdxl\` | [ByteDance SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_8step_lora.safetensors) |
| `xinsir_controlnet-tile-sdxl-1.0.safetensors` | `models\controlnet\sdxl\` | source file 是 [xinsir/controlnet-tile-sdxl-1.0/diffusion_pytorch_model.safetensors](https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors)，下載後 rename 成 workflow filename；SHA256 `9f23ba7be22bf8796c12565e00ea4b287acac982cdf384d368a8b18b6990e011` |
| `Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors` | `models\controlnet\` | [Alibaba-PAI model file](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors) |
| `depth_anything_v2_vitl.pth` | `custom_nodes\comfyui_controlnet_aux\ckpts\depth-anything\Depth-Anything-V2-Large\` | [Depth Anything V2 Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_vitl.pth)；目前 machine 已有 1,341,395,338 bytes |
| `Baked VAE` | 由 Tile workflow 的 checkpoint loader 選擇 | 不是獨立 model URL；依 workflow dropdown / checkpoint 內容，不要另造檔案 |

`nova-anime-xl.json` 會同時需要兩個 Nova checkpoint、WAI checkpoint 與 `reiXL_NB11` LoRA；只裝其中三個不能宣稱該 workflow ready。`z_image_plus_nova_model.json` 仍是「Z-Image + Nova」混合 graph，不是 pure Z-Image graph。

### 5.6 Workflow model settings 與目前保留的 optimization

安裝 model 不會自動套用這些 graph 參數；參數來自 JSON，runtime binding 只替換 prompt、seed、input path 等允許欄位。LLM 或安裝者不得為了排除缺 model 而改 sampler、denoise 或 LoRA strength。

| Workflow / stage | 目前設定 |
|---|---|
| krea2_turbo | 1024×576 smoke test、batch 1、8 steps、CFG 1、Euler/simple、denoise 1.0；negative path 使用 ConditioningZeroOut；prompt enhancer disabled |
| krea2_turbo_img2img | 同一組 Krea/Qwen/VAE，8 steps、CFG 1、Euler/simple、denoise 0.25；用於 continuity repair |
| anima_anime | 1024×1024、batch 1、25 steps、CFG 3.5、DPM++ 2M SDE GPU、SGM uniform、denoise 1.0；輸出後使用 4x-AnimeSharp |
| kirby_keyframe_anima | 608×352、batch 1、25 steps、CFG 3.5、DPM++ 2M SDE GPU、SGM uniform、denoise 1.0；Anima CLIP device default |
| kirby_identity_img2img | Anima、15 steps、CFG 3.5、DPM++ 2M SDE GPU、SGM uniform、denoise 0.35；CLIP device CPU |
| kirby_continuity_img2img | Anima、15 steps、CFG 3.5、DPM++ 2M SDE GPU、SGM uniform、denoise 0.18；CLIP device CPU |
| image_to_image | WAI Illustrious SDXL、25 steps、CFG 8、Euler/normal、denoise 0.7 |
| nova-anime-xl | Nova/WAI SDXL、25 steps、CFG 8、Euler/normal、denoise 1.0；reiXL LoRA strength_model 0.8、strength_clip 0.85；無 LoRA 就不是同一 graph |
| nova_model_plus_z_image_anime | Z-Image stage 9 steps、CFG 1、res_multistep/simple；Nova/SDXL stage 20 steps、CFG 8、Euler/normal；Depth Anything preprocessor resolution 512 |
| z_image_plus_nova_model | Z-Image stage 9 steps、CFG 1、res_multistep/simple；SDXL refinement 25 steps、CFG 8、Euler/normal、denoise 0.9 |
| Tile Upscaler SDXL path A | 512×512、batch 1、SDXL Lightning LoRA model/CLIP strength 1、4 steps、CFG 1、Euler/SGM uniform、denoise 0.6；Tile ControlNet strength 0.5 |
| Tile Upscaler SDXL path B | 4 steps、CFG 2、Euler/DDIM uniform、denoise 0.5；Tile ControlNet strength 0.7；Essentials tiling、WAS sharpening、Impact list conversion |
| H3 low-VRAM T2V/I2V | 608×352、24 fps、124 frames、20 steps、res_multistep；shift_video 12、shift_audio 3、history_storage system_ram、Spectrum enabled |
| H3 15s FL2VA | 608×352、24 fps、362 frames、16 steps；first/last-frame conditioning 依 workflow，不可用一般 I2V graph 代替 |
| H3 Ref2VA | 608×352、24 fps、124 frames、20 steps；reference image/video path、1–4 references；Spectrum 不使用 |

Krea→H3 的顯存 optimization 是「Krea 產圖後呼叫 /free，再載入 H3」，不是永久把兩套 model 同時留在 GPU。H3 launcher 的保守旗標、Q4/Q2 profile、Spectrum system RAM 與 608×352 是目前 RTX 4060 8 GB baseline；提高解析度、length、batch 或改 DynamicMode 都應建立新的 smoke-test evidence。

## 6. 下載 model 的可靠方式

### 6.1 PowerShell 下載 helper

使用 `curl.exe` 而不是 PowerShell 的 `curl` alias；`--continue-at -` 允許中斷後續傳，`--retry` 避免短暫網路錯誤。下載完成後才把檔案放進正式目錄，避免 ComfyUI 讀到半個檔案：

```powershell
function Download-ComfyModel {
    param(
        [Parameter(Mandatory=$true)][string]$Url,
        [Parameter(Mandatory=$true)][string]$Target
    )
    $Parent = Split-Path -Parent $Target
    New-Item -ItemType Directory -Force -Path $Parent | Out-Null
    $Part = "$Target.part"
    curl.exe --location --fail --retry 5 --retry-delay 5 --continue-at - --output $Part $Url
    if ($LASTEXITCODE -ne 0) { throw "download failed: $Url" }
    Move-Item -Force -LiteralPath $Part -Destination $Target
}

function Assert-ComfyModel {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][long]$Bytes,
        [Parameter(Mandatory=$true)][string]$Sha256
    )
    $Item = Get-Item -LiteralPath $Path
    if ($Item.Length -ne $Bytes) { throw "size mismatch: $Path ($($Item.Length) != $Bytes)" }
    $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($Hash -ne $Sha256.ToLowerInvariant()) { throw "sha256 mismatch: $Path ($Hash != $Sha256)" }
    "OK $Path $Bytes $Hash"
}
```

### 6.2 下載 Krea 2

```powershell
$KreaUnet = Join-Path $Comfy 'models\unet\krea2_turbo_bf16-Q4_0.gguf'
$KreaClip = Join-Path $Comfy 'models\clip\Qwen3VL-4B-Instruct-Q4_K_M.gguf'
$KreaVae = Join-Path $Comfy 'models\vae\qwen_image_vae.safetensors'

Download-ComfyModel 'https://huggingface.co/molbal/krea2-gguf/resolve/main/krea2_turbo_bf16-Q4_0.gguf' $KreaUnet
Download-ComfyModel 'https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen3VL-4B-Instruct-Q4_K_M.gguf' $KreaClip
Download-ComfyModel 'https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors' $KreaVae

Assert-ComfyModel $KreaUnet 8314016800 '3d28960e1f3385b27a224c6968a0acc02b07194758247e576ee0c5caaab4f1cf'
Assert-ComfyModel $KreaClip 2497281664 '66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a'
Assert-ComfyModel $KreaVae 253806246 'a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f'
```

### 6.3 使用 repo H3 installer

不要手動改檔名。這支 script 會依 profile 放到正確目錄，下載中斷時保留 `.part`，並檢查 expected bytes：

```powershell
Set-Location $Repo

& $ComfyPython $Repo\scripts\setup_minimax_h3.py --profile balanced-lowvram --comfy-root $ComfyRoot --json
& $ComfyPython $Repo\scripts\setup_minimax_h3.py --profile ref2va-lowvram --comfy-root $ComfyRoot --json
& $ComfyPython $Repo\scripts\setup_minimax_h3.py --profile ultra-lowvram --comfy-root $ComfyRoot --json
& $ComfyPython $Repo\scripts\setup_minimax_h3.py --status --comfy-root $ComfyRoot
```

`ref2va-lowvram` 會重用已經下載的兩個 VAE 與 Q4 text；不會因為再次執行而重複下載完整檔案。若只要主線 T2V/I2V，前兩個 profile 即可；`ultra-lowvram` 只是在已需要 Q2 fallback 時確認 Q2 text。

原生 profile 是明確的額外選項：

```powershell
& $ComfyPython $Repo\scripts\setup_minimax_h3.py --profile native-quality --comfy-root $ComfyRoot --json
```

這一步會下載約 39.55 GiB 的 native weights，不是必要 smoke test；開始前先確認磁碟與 GPU/RAM 預算。

### 6.4 下載 Anima / SDXL optional assets

只有在要啟用相應 workflow 時才下載。PowerShell 範例：

```powershell
# Anima
Download-ComfyModel 'https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-aesthetic-v1.1.safetensors' (Join-Path $Comfy 'models\diffusion_models\anima\anima-aesthetic-v1.1.safetensors')
Download-ComfyModel 'https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors' (Join-Path $Comfy 'models\text_encoders\anima\qwen_3_06b_base.safetensors')
Download-ComfyModel 'https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors' (Join-Path $Comfy 'models\vae\anima\qwen_image_vae.safetensors')
Download-ComfyModel 'https://huggingface.co/Kim2091/AnimeSharp/resolve/main/4x-AnimeSharp.pth' (Join-Path $Comfy 'models\upscale_models\4x-AnimeSharp.pth')

# Z-Image / ControlNet
Download-ComfyModel 'https://huggingface.co/torestinbar/z-image-turbo/resolve/main/checkpoints/zImageTurboAnime_aioBF16.safetensors' (Join-Path $Comfy 'models\checkpoints\z_image\zImageTurboAnime_aioBF16.safetensors')
Download-ComfyModel 'https://huggingface.co/Toc/toc/resolve/main/models/noobaiXLNAIXL_vPred10Version.safetensors' (Join-Path $Comfy 'models\checkpoints\sdxl\noobaiXLNAIXL_vPred10Version.safetensors')
Download-ComfyModel 'https://huggingface.co/Comfy-Org/z_image/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors' (Join-Path $Comfy 'models\text_encoders\qwen\qwen_3_4b.safetensors')
Download-ComfyModel 'https://huggingface.co/Comfy-Org/z_image/resolve/main/split_files/vae/ae.safetensors' (Join-Path $Comfy 'models\vae\flux\ae.safetensors')
Download-ComfyModel 'https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors' (Join-Path $Comfy 'models\diffusion_models\z_image\z_image_turbo_bf16.safetensors')
Download-ComfyModel 'https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors' (Join-Path $Comfy 'models\controlnet\Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors')
Download-ComfyModel 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth' (Join-Path $NodeRoot 'comfyui_controlnet_aux\ckpts\depth-anything\Depth-Anything-V2-Large\depth_anything_v2_vitl.pth')

# Tile ControlNet: source filename must be renamed to the filename used by JSON.
Download-ComfyModel 'https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors' (Join-Path $Comfy 'models\controlnet\sdxl\xinsir_controlnet-tile-sdxl-1.0.safetensors')
```

上面的 Z-Image optional example 依照 Comfy-Org split-file layout；如果目前 branch 的 JSON 仍使用其他 loader/model path，以 JSON 的 `class_type` 與 `*_name` 為最後準則。下載後不要只看檔名，要執行第 8 節的 workflow asset scan。

## 7. ComfyUI 的啟動設定與 VRAM 優化

### 7.1 repo 目前的正式 H3 launcher

在 repo root 執行：

```powershell
Set-Location $Repo
& powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_comfyui_h3_lowvram.ps1 `
    -ComfyRoot $ComfyRoot `
    -Port 8188
```

預設啟動參數是：

```text
--windows-standalone-build
--disable-auto-launch
--listen 127.0.0.1
--port 8188
--reserve-vram 1.0
--disable-pinned-memory
--disable-dynamic-vram
--disable-async-offload
--lowvram
```

這些設定是針對目前 H3 low-VRAM workflow 的保守起點：避免 dynamic VRAM/offload 在 8 GB card 上增加不穩定性；Spectrum graph 的 `history_storage=system_ram` 也要保留。`-FastDisk` 只在確認 Windows 磁碟與顯存行為後使用；`-DynamicMode` 是 A/B 測試選項，不是預設值。

目前工作機的 live process 另外帶有 --enable-manager，那是維護介面選項，不是 workflow graph dependency；repo launcher 的基線不依賴它。若要重現目前觀察到的 Manager-enabled argv，請在同一個 low-VRAM launcher 的 argument list 中額外加入 --enable-manager，並把這個選擇寫入 machine manifest。

### 7.2 Krea → H3 的 model lifecycle

Krea Q4 可能佔用 8 GB GPU 大部分顯存。Krea image 完成且要載入 H3 前，必須釋放 Krea：

```powershell
Invoke-RestMethod -Method Post `
    -Uri 'http://127.0.0.1:8188/free' `
    -ContentType 'application/json' `
    -Body '{"unload_models":true,"free_memory":true}'
```

repo adapter 已經把這個 lifecycle boundary 納入；直接寫 API smoke test 時也要做。這是避免「Krea 仍在 GPU cache，H3 text encoder OOM」的必要步驟，不是清理輸出檔案。

### 7.3 Port / Docker 設定

本機直接跑 repo：

```dotenv
COMFYUI_API_URL=http://127.0.0.1:8188
```

Docker scheduler 連 Windows host：

```dotenv
SCHEDULER_COMFY_HOST=host.docker.internal
SCHEDULER_COMFY_PORT=8188
SCHEDULER_COMFY_ROOT=/comfyui
```

`docker-compose.yml` 預期宿主機 portable 目錄是 `D:/ComfyUI_windows_portable:/comfyui:ro`。如果你把 ComfyUI 放在別的磁碟，要同步修改 compose volume；容器內的 `/comfyui` 不能寫成 Windows `D:\...`。

## 8. 安裝完成驗證：每一關都要有 evidence

### Gate A：server / version / device

```powershell
$Stats = Invoke-RestMethod 'http://127.0.0.1:8188/system_stats'
$Stats.system | Format-List
```

必須確認：

- server HTTP 可達；
- `comfyui_version` 是選定版本；
- `python_version` 是 embedded Python；
- `pytorch_version` 是 NVIDIA CUDA build；
- device 是預期 GPU；
- port 是 8188，沒有被另一個 ComfyUI process 搶走。

### Gate B：node class

```powershell
$ObjectInfo = Invoke-RestMethod 'http://127.0.0.1:8188/object_info'
$RequiredClasses = @(
    'UnetLoaderGGUF','CLIPLoaderGGUF','MiniMaxH3ImageToVideo',
    'MiniMaxH3ReferenceToVideo','MiniMaxH3SigmaShift','SpectrumApplyMiniMaxH3',
    'SaveVideo','CreateVideo','VHS_LoadVideoPath',
    'DepthAnythingV2Preprocessor','ImageTile+','ImageUntile+',
    'ImageListToBatch+','Image Comparer (rgthree)','Image Lucy Sharpen',
    'ImpactImageBatchToImageList','ACN_ReferenceControlNet',
    'ACN_ReferencePreprocessor'
)
foreach ($Class in $RequiredClasses) {
    if ($null -eq $ObjectInfo.$Class) { throw "Missing node class: $Class" }
    "FOUND $Class"
}
```

### Gate C：model size/hash/path

```powershell
Get-ChildItem (Join-Path $Comfy 'models') -Recurse -File |
    Where-Object { $_.Length -gt 0 } |
    Select-Object FullName, Length |
    Sort-Object FullName

& $ComfyPython $Repo\scripts\setup_minimax_h3.py --status --comfy-root $ComfyRoot
```

Krea 三個檔案必須通過第 6.2 節的 bytes + SHA；H3 至少要通過 profile 的 expected bytes。不能只用 `Test-Path`，因為中斷下載留下的檔案也可能存在。

### Gate D：workflow references scan

逐一檢查 `configs/workflow/*.json`：

1. 讀每個 node 的 `class_type`，確認 class 出現在 `/object_info`。
2. 讀 `unet_name`, `clip_name`, `vae_name`, `ckpt_name`, `lora_name`, `control_net_name`, `model_name`。
3. 將反斜線轉成 path separator，對應到 `ComfyUI\models` 或 custom node checkpoint 目錄。
4. 每一個 reference 都要是 `FOUND + nonzero + verified size/hash`；否則 workflow status 是 `NOT_READY`。

目前 audit 結果：Krea、H3 low-VRAM、H3 Ref2VA 的 references FOUND；native H3、Anima、SDXL/Nova/Tile 有缺失，不能在乾淨機器上宣稱全部 route ready。

### Gate E：最小 live smoke test 順序

不要一開始跑完整排程或多角色批次。順序固定如下：

1. Krea 2 T2I：1024×576，單張，8 steps；確認得到 nonzero PNG。
2. 呼叫 `/free`，確認 Krea model unload。
3. H3 low-VRAM T2V：608×352，124 frames，24 fps，20 steps；確認得到 nonzero MP4。
4. Krea approved still → H3 I2V：先做 image input，再驗證 H3 output。
5. 最後才做 15 秒 FL2VA 或 Ref2VA；Ref2VA 必須先有有效 reference image/video path。

每一關都要保存：workflow name、ComfyUI version、node commit、model filename、model SHA/size、啟動 argv、output path、duration、是否 OOM。只寫「成功」不足以重建或診斷。

## 9. Repo app 環境與 ComfyUI 環境的分界

ComfyUI embedded Python 與 repo `.venv` 是兩個 runtime：

```powershell
Set-Location $Repo
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
& .\.venv\Scripts\python.exe -m pip install -e .\agentic
```

repo app 的 `requirements.txt` / `agentic/pyproject.toml` 主要是 scheduler、LLM、DB、Discord、Google、MoviePy 等；custom node 的 requirements 才裝進 ComfyUI embedded Python。兩者都裝到同一個 Python 會導致版本互相污染。

建立環境檔：

```powershell
Copy-Item .\media_overload.env.example .\media_overload.env
```

至少確認：

- `AGENTIC_TEXT_MODEL_PROVIDER=openrouter` 與 OpenRouter key/model pool；
- 本機 ComfyUI 使用 `127.0.0.1:8188`；
- Docker 使用 `host.docker.internal:8188`；
- DB、Discord、Twitter/Google 等 token 是另外的外部依賴；
- Ollama 設定可以保留，但不能把「repo 有 Ollama fallback 設定」誤判成 ComfyUI 已裝好 native/Ollama model。

## 10. 常見問題與判斷順序

| 症狀 | 先查 | 正確處理 |
|---|---|---|
| `Cannot import ...` / node class missing | ComfyUI console、`custom_nodes` git commit、該 node requirements | 先修 package/import；不要先重下載 model |
| `UnetLoaderGGUF` 不存在 | ComfyUI-GGUF 是否在 active directory，`gguf` 是否裝入 embedded Python | 重裝 GGUF node requirements，重啟 server |
| `MiniMaxH3...` 不存在 | ComfyUI core 版本與 H3 core class | 使用已驗證 ComfyUI source；不要用舊版 portable |
| `Model ... not found` | JSON loader name 與 `ComfyUI\models` 相對路徑 | 檔名、子目錄、loader 類型三者要完全一致 |
| 檔案存在但載入失敗 | bytes/SHA、是否仍有 `.part`、磁碟是否滿 | 刪除不完整 `.part`，重新下載並驗證 |
| Krea → H3 OOM | 是否先呼叫 `/free`；是否開了 dynamic mode | unload Krea，使用 low-VRAM launcher；先不要提高解析度 |
| 15 秒 H3 OOM | length 362、resolution、text model、history storage | 先以 124-frame profile 通過，再測 15 秒；保留 `system_ram` |
| Ref2VA 找不到影片 | VideoHelperSuite 與 `VHS_LoadVideoPath` | 安裝 node，確認 path 是容器/宿主機可見的正確 path |
| Docker 連不到 ComfyUI | `host.docker.internal`、Windows firewall、port 8188 | 宿主機 listen 127.0.0.1 與 Docker networking 必須一起驗證；必要時依 compose 設定調整 |
| Manager 顯示可安裝但 workflow 仍失敗 | Manager model/node catalog 不等於 repo manifest | 以本手冊與 JSON asset scan 為準，記錄實際 commit/hash |
| native profile 失敗 | `models\diffusion_models` / `text_encoders`、VRAM/RAM | native 是額外 profile；回到 GGUF Q4/Q2 基線 |
| Nova route 失敗 | 四個 assets 是否全存在 | `reiXL_NB11` 來源目前 BLOCKED；不要用另一個 LoRA 冒充 |

## 11. 完成安裝時應產生的 machine manifest

全新機器完成後，請把以下輸出保存到 repo 外的安裝紀錄；如果要讓其他人重建，應將不含 secret 的版本提交到專案：

```powershell
$Manifest = Join-Path $Repo 'comfyui-install-manifest.txt'
@(
    "captured_at=$(Get-Date -Format o)"
    "comfy_root=$ComfyRoot"
    "comfy_version=$((Invoke-RestMethod 'http://127.0.0.1:8188/system_stats').system.comfyui_version)"
    "python=$(& $ComfyPython --version 2>&1)"
    "torch=$(& $ComfyPython -c 'import torch; print(torch.__version__)')"
    "cuda=$(& $ComfyPython -c 'import torch; print(torch.version.cuda)')"
    "gpu=$(& $ComfyPython -c 'import torch; print(torch.cuda.get_device_name(0))')"
) | Set-Content -Encoding utf8 $Manifest

Get-ChildItem $NodeRoot -Directory | ForEach-Object {
    if (Test-Path (Join-Path $_.FullName '.git')) {
        Add-Content $Manifest "node=$($_.Name) remote=$(git -C $_.FullName remote get-url origin) commit=$(git -C $_.FullName rev-parse HEAD)"
    }
}

Get-ChildItem (Join-Path $Comfy 'models') -Recurse -File |
    Sort-Object FullName |
    ForEach-Object {
        $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
        Add-Content $Manifest "model=$($_.FullName) bytes=$($_.Length) sha256=$Hash"
    }
```

不要把 API key、DB password、OAuth token 寫進 manifest。manifest 的目的，是把「哪個 ComfyUI source / node commit / model bytes / SHA」與 secret 分離保存。

## 12. LLM 操作契約

任何之後協助安裝或修復的 LLM，必須依以下順序工作：

1. 先讀本文件、`configs/routing.yaml`、目標 workflow JSON、`minimax_h3.py`、`workflow_config.yaml`。
2. 先查 `system_stats` 與 `object_info`，再碰 model 或 plugin。
3. 把問題分類為 `server/version`、`node/package`、`model/path/hash`、`VRAM/lifecycle`、`repo app/external service` 五類之一。
4. 不得把 model filename 相同的第三方 mirror 當成同一 model；沒有 SHA 就標成未驗證。
5. 不得自行替換 `reiXL_NB11`、native H3 或其他缺來源 model。
6. 任何「已安裝」結論都必須附：絕對路徑、檔案大小、SHA（若有）、node commit、ComfyUI version、實際 live endpoint evidence。
7. 修改 workflow 前，先確認是使用者要改 graph，還是只是安裝/環境問題；安裝任務不能順手改 routing probability、prompt、character config。

## 13. 最終 ready / not ready 判定

### `READY: mainline-lowvram`

只有在以下全部成立時才可使用：

- ComfyUI server 可達，version 與 Python/torch/GPU 已記錄；
- GGUF、Spectrum、VideoHelperSuite node class 已載入；
- Krea 三個 model 完成 bytes + SHA；
- H3 balanced 或 ultra profile 完成 expected bytes；
- H3 Ref2VA 所需 model 與 VideoHelperSuite 完成；
- Krea T2I、`/free`、H3 124-frame smoke test 都產生有效 artifact；
- output path 與 repo scheduler 的 host/root 設定一致。

### `NOT_READY: all-repo-workflows`

只要 `reiXL_NB11` 尚未取得原始 URL + SHA、或任一 workflow 仍有 missing node/model，就不能使用這個標籤。這不是安裝失敗，而是 repo 目前的 source-of-truth 還沒有覆蓋所有歷史 workflow；應先補 manifest/lock，再宣稱 full reproducibility。

## 14. 建議的 repo 後續修補

為了讓未來真的達到「全新 Windows 電腦無痛安裝」，下一個工程變更應該：

1. 把所有 workflow 的 model reference 收斂到一份 machine-readable manifest，包含 URL、revision、bytes、SHA256、target path、license/source type。
2. 把沒有 commit 的 custom node 變成 pinned commit，或把可驗證 snapshot 放到正式 release artifact。
3. 為 `reiXL_NB11` 補原始 model page、model/version ID、license、SHA；在此之前將依賴它的 Nova route 明確標為 unavailable。
4. 增加一個不需要 DB/LLM/provider 的 `verify_comfyui_install.ps1`，自動執行 Gate A–D；live render 另做 Gate E，避免把外部服務誤混入 ComfyUI preflight。

這四項完成後，手冊中的 `BLOCKED` 與 `URL NOT RECORDED` 才能移除；在此之前保留這些標記，比用未驗證 model 造成錯誤的身份/畫質結論更安全。
