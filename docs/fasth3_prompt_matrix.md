# FastH3 multi-prompt quality matrix

日期：2026-09-18（Asia/Taipei）  
分支：`codex/fasth3-benchmark`  
GPU：NVIDIA GeForce RTX 4060 8 GiB

## 測試設計

這是對單一 Kirby 測試的 follow-up。這次使用三個完全不同的 prompt；每個 prompt 都用相同 seed 分別跑現有 GGUF、official native INT8 與 FastH3 V2 INT8 + VSA。每個 cell 都是獨立的 ComfyUI E2E 影片，包含模型準備、sampling、video/audio VAE decode、CreateVideo、SaveVideo；不是重用前一支影片。

| Prompt | 測試重點 | Seed |
|---|---|---:|
| `fox_scarf` | 狐狸追逐飛向冰溪的圍巾，跳起接住，再圍到脖子上 | 20260919 |
| `paper_boat` | 紙船接近排水孔，鳥用樹枝移開障礙，船進入平靜水面、花朵綻放 | 20260920 |
| `robot_flower` | 機器人注意到枯萎花朵，拿水壺澆水，花朵恢復並開花 | 20260921 |

品質檢視素材：`E:\comfyui\_extra\benchmarks\fasth3_matrix_quality`。每支影片以 `ffmpeg` 每 0.5 秒抽幀，再按 prompt 的時間區間檢查事件順序、物件一致性、角色動作與 payoff。

## E2E 結果

| Prompt | GGUF baseline | Native INT8 | FastH3 V2 INT8 + VSA |
|---|---:|---:|---:|
| `fox_scarf` | 286.614s / 7913 MiB | 509.390s / 7570 MiB | 282.169s / 7541 MiB |
| `paper_boat` | 280.165s / 7899 MiB | 385.799s / 7498 MiB | 274.038s / 7441 MiB |
| `robot_flower` | 281.880s / 7890 MiB | 372.199s / 7439 MiB | 272.953s / 7574 MiB |
| **平均** | **282.886s** | **422.463s** | **276.387s** |

9/9 個 E2E runs 通過。所有輸出都是 `608x352`、`124 frames`、`24 fps`、`5.166667s`、H.264 video、AAC stereo 32 kHz。

速度只代表這批 cold/uncached run：FastH3 平均比 GGUF 快約 2.3%（6.499 秒），native INT8 平均慢約 49.3%。FastH3 的速度優勢在這台機器上是小幅度，不應誇大成穩定 SLA。

## 語意與畫面品質判定

### `fox_scarf`

- GGUF 能呈現雪地、圍巾、狐狸與暖光，但狐狸的「跳起接住」較像靠近後拿到。
- Native INT8 的圍巾有大幅度近景與形狀變化，畫面細節不錯，但事件節點較難讀。
- FastH3 V2 的狐狸跳過冰面、圍巾飛行、最後圍到脖子上的連續事件最容易讀；這題 FastH3 的時間語意勝出，但圍巾比例仍偏大，不能算完全穩定。

### `paper_boat`

- GGUF 與 native 都保留紅船、黃花、藍鳥與雨景，但「鳥移開樹枝、讓船避開排水孔」不明確。
- FastH3 V2 明確呈現排水孔危機、鳥與樹枝互動、船離開危險位置並進入有睡蓮的平靜水面；這題是三者中 FastH3 語意優勢最清楚的一題。
- Native INT8 的花草與水面細節漂亮，但視覺細節沒有轉化成更好的因果敘事。

### `robot_flower`

- GGUF 的機器人、澆水壺與花朵很穩，澆水動作清楚，但花朵從枯萎到開花的變化較弱。
- Native INT8 能讀到機器人澆水與花朵逐漸打開，畫面品質穩定。
- FastH3 V2 的角色接近、拿壺、出水、花朵由花苞變成盛開的因果鏈最完整；這題 FastH3 與 native 都明顯優於原本的單一 Kirby 結論。

## 更新後結論

多 prompt 結果修正了單一 Kirby 測試的保守結論：**FastH3 V2 不是單純的畫質升級，但在這三個新的動作／因果 prompt 上，時間語意與 prompt adherence 整體比 GGUF 更好，native INT8 則偏向單張畫面細節與穩定性。**

因此建議：

1. 不把 FastH3 V2 直接替換成所有 H3 route 的 default；它仍有特定物件比例與動作一致性風險。
2. 將 FastH3 V2 保留為短 T2AV 的候選 production route，特別是需要清楚動作順序、物件互動與結尾變化的 prompt。
3. Native INT8 不適合目前 8 GiB GPU 作為速度優化 route；它平均最慢，品質優勢主要是細節，不是語意。
4. 正式切 production default 前，再用至少 10 個 prompt、每題 2 個 seed，做盲測與人審；本輪 3 題已足以證明 FastH3 值得保留，但還不足以宣稱普遍勝出。

### 抽幀對照

![fox scarf route comparison](E:/comfyui/_extra/benchmarks/fasth3_matrix_quality/fox_scarf_routes_halfsec.jpg)

![paper boat route comparison](E:/comfyui/_extra/benchmarks/fasth3_matrix_quality/paper_boat_routes_halfsec.jpg)

![robot flower route comparison](E:/comfyui/_extra/benchmarks/fasth3_matrix_quality/robot_flower_routes_halfsec.jpg)
