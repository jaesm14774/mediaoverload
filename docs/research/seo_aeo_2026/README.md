# 2026 SEO / AEO / 社群推薦：MediaOverload 深度優化

研究日期：2026-09-02

這份研究的結論不是「找到一個能騙過所有平台的格式」。2026 年的可見度要拆成兩條鏈：

1. 搜尋鏈：頁面或貼文是否可被抓取、理解、引用，且是否提供原創、可驗證、能解決問題的內容。
2. 推薦鏈：觀眾是否願意停留、看完、分享、回訪，並在不同推薦 surface 上得到滿意結果。

SEO、AEO、GEO 是不同社群對同一組搜尋可理解性工作的命名；Google 的說法是，生成式 AI Search 仍使用搜尋索引與既有 SEO 基礎，不需要一套神奇的 `llms.txt` 或特殊 AI 標記。社群推薦演算法則不只看關鍵字，也會看內容吸引力、觀看／互動品質、滿意度、原創性與政策資格。

## 一、來源形成的可執行結論

| 研究結論 | 對貼文的實際要求 | 不採用的做法 |
| --- | --- | --- |
| People-first、原創與第一手價值比搜尋引擎優先寫法重要 | 先說清楚媒體中真正可見的主體、動作、變化與觀察 | 批量改寫熱門題目、填充關鍵字、固定文章模板 |
| AI Search 仍依賴可索引、可理解的內容 | 將來源 topic、visible evidence、因果 bridge 留在可追蹤資料與可讀文字中 | 為每一個 fan-out query 產生一篇薄頁面 |
| 生成式答案的被引用與被吸收是兩個階段 | 用具體定義、數字、比較、程序或可查證來源支持真正的 claim | 只把句子改成 Q&A 就宣稱 AEO 完成 |
| YouTube recommendation 是 appeal、engagement、satisfaction 多訊號系統 | 測試首段吸引力、觀看留存、分享、回訪與滿意回饋 | 以日更頻率或單一觀看數當成全部目標 |
| Meta / Instagram 會抑制 spammy、無關、過度 hashtag 或 fake engagement | hashtag 只保留對本則內容有語意幫助的少數標籤，甚至為零 | 固定 `#FYP`、`#ForYou`、內部專案標籤或每篇塞滿標籤 |
| 推薦系統若只追逐短期興趣，容易連續推薦同質內容 | 同一角色／主題輪換觀察角度與視覺 payoff，保持內容家族但不複製文案 | 每篇只換形容詞、emoji 或標籤 |

## 二、這個專案採用的貼文模型

每次 `prepare-caption` 先建立一份 editorial variation brief。它只回答「這次值得從哪個角度看」，不規定句數、段落數、CTA、emoji、語氣或文案長度。

目前的角度池：

- `visible_moment`：單一可見的變化或意外。
- `cause_and_effect`：一個被畫面支持的動作與結果。
- `character_choice`：主角的可見選擇、反應或取捨。
- `replay_detail`：值得重看的一個具體細節。
- `contrast`：只有在兩個狀態都看得到時，才使用前後／尺度／色彩／動勢對比。
- `news_mechanism_bridge`：把已驗證的新聞機制連到視覺隱喻，並明確不把動畫冒充成真實事故證據。

角度由 goal、媒體 basename、新聞 context 與 `post_strategy_seed` 的 hash 穩定選出，並記錄 `variation_key`。一般 character workflow 會把每次 workflow 的 `run_id` 傳入 seed，因此同一 run 可重現、不同 run 可輪換；直接呼叫 publish flow 時也可用 media basename/context 形成穩定結果。若人類在 Discord 核准或編輯了文字，核准文字仍是唯一的 publish copy，brief 只留作實驗與稽核 metadata，不會回頭改寫文字。

程式契約位於：

- `agentic/src/agentic/runtime/post_strategy.py`
- `agentic/src/agentic/runtime/llm_engine.py`
- `agentic/src/agentic/runtime/platform_content.py`
- `agentic/src/agentic/skills/agent_social.py`

## 三、hashtag 與搜尋語意規則

### hashtag

1. 模型可以回傳零到三個 hashtag；空字串是合法結果。
2. config 傳入的 tags 是 hints，不是必帶清單；只有模型自己選出且通過內容語意的 tag 才會留下。
3. 每個 tag 必須對應可見主體、可見動作／場景、或明確的來源 topic。
4. 沒有額外 discovery 價值時，直接不放 hashtag。
5. 不用 `#FYP`、`#ForYou` 或同義 reach-bait 填空間；也不使用 `#mediaoverload` 這類內部名稱。
6. Facebook 的公開包裝最多三個 tag；這是平台欄位控制，不代表每篇都要湊三個。

### title、caption、description

- 文案第一句可以自然地承載最重要的可見 claim，YouTube title／description 再由已核准或已生成的 copy 衍生。
- 不從整個 generation prompt 無差別抽 keyword；prompt 可能包含 production metadata 或不在畫面中的內容。
- `discovery_terms` 只記錄新聞 context 明確提供的 topic、title、keywords、entities，不自行發明關鍵字。
- SEO/AEO 可理解性不是「加更多關鍵字」，而是讓一個真實讀者、搜尋引擎與答案引擎都能辨識：這段內容在談什麼、證據在哪裡、限制是什麼。

## 四、接到現有發文標準流程

目前流程仍是：

```text
ingest-media
  -> review-select（人類 Discord 審核）
  -> process-media
  -> prepare-caption（strategy brief + visual grounding + optional hashtags）
  -> dispatch-publish（platform bundle + live adapter）
  -> persist-publish-review-summary
```

各節點責任如下：

| 節點 | 新增或保留的責任 |
| --- | --- |
| `review-select` | 人類決定媒體與主觀內容品質，仍是最高優先權。 |
| `prepare-caption` | 未核准時，把 brief、視覺證據、新聞 grounding contract 提供給模型；不加 paragraph／takeaway／question gate。 |
| `platform_content` | 將同一個已核准 claim 做 YouTube / Facebook 等欄位包裝；記錄角度、版本、variation key、hashtag policy 與 source terms。 |
| `dispatch-publish` | 只負責資格、媒體、平台 adapter 與 receipt；本地 bundle 成功不等於社群已上線。 |
| `persist-publish-review-summary` | 保留發文前 copy、策略 metadata、平台 bundle 與後續 receipt，供實驗回溯。 |

這個邊界很重要：AI 可以在人工核准前幫忙提供多角度草稿；人工核准後不再跑第二次 LLM 內容 gate，也不把平台包裝誤報成 live publishing。

## 五、建議的實驗與判讀方式

不要用「文案長得像 SEO」當成成功標準。以同一角色／內容族群，在相近媒體品質與發布時段下，記錄：

- 內容層：`variant_id`、`variation_key`、是否使用 hashtag、來源 terms、是否有 news bridge。
- YouTube：impressions CTR、audience retention、returning viewers、可取得的 satisfaction feedback。
- Facebook / Instagram：qualified views、average watch time、分享、收藏、meaningful comments、profile actions；實際可用欄位依 API 與帳號權限為準。
- 搜尋層：Search Console 的 impressions、clicks、video／social appearance；若使用第三方 AEO/GEO monitor，需標示為外部量測，不冒充 Google 內部排名訊號。

每輪只改一個主要變因，例如「角度」或「hashtag 有／無」，不要同時改 caption、媒體、時段與平台。研究結果應保留不確定性：平台 ranking model 會更新，離線論文與 repo benchmark 不能直接推導某支影片一定增加觀看數。

## 六、研究與 repo 審查

### 官方平台與搜尋文件

- [Google Search Essentials](https://developers.google.com/search/docs/essentials)
- [Google：Creating helpful, reliable, people-first content](https://developers.google.com/search/docs/fundamentals/creating-helpful-content)
- [Google：AI features and your website](https://developers.google.com/search/docs/fundamentals/ai-optimization-guide)
- [Google：Using generative AI content on your website](https://developers.google.com/search/docs/fundamentals/using-gen-ai-content)
- [Google spam policies](https://developers.google.com/search/docs/essentials/spam-policies)
- [Google：Video SEO](https://developers.google.com/search/docs/appearance/video)
- [Google Search Console：social and video performance](https://developers.google.com/search/docs/monitor-debug/analyze-social-video-content)
- [YouTube recommendation system](https://support.google.com/youtube/answer/16089387)
- [YouTube performance FAQ](https://support.google.com/youtube/answer/141805)
- [YouTube recommendation performance](https://support.google.com/youtube/answer/16559650)
- [Meta：Cracking down on spammy content](https://about.fb.com/news/2025/04/cracking-down-spammy-content-facebook/)
- [Meta：Recommendation Guidelines](https://about.fb.com/news/2020/08/recommendation-guidelines/)
- [Instagram：How Instagram ranking works](https://about.instagram.com/blog/announcements/instagram-ranking-explained)
- [TikTok：How TikTok recommends content](https://support.tiktok.com/en/using-tiktok/exploring-videos/how-tiktok-recommends-content)
- [TikTok creator tips](https://newsroom.tiktok.com/5-tips-for-tiktok-creators?lang=en)

Instagram 的 ranking 說明頁是官方且有用的機制說明，但不是 2026 年完整公式；應當視為公開原則，不是永遠不變的權重表。

### 文獻與開源 repo

- [GEO: Generative Engine Optimization, KDD 2024](https://doi.org/10.1145/3637528.3671900)：顯示 citation、quotes、statistics 等內容特徵可能改變生成式答案中的可見度；不等於社群觀看保證。
- [Citation Selection and Absorption in Generative Engines, 2026 preprint](https://arxiv.org/abs/2604.25707)：把「被選中」與「被答案吸收」分開，觀察到較有結構、定義、數字、比較與程序的 evidence containers；作者也明確提醒是 descriptive、非因果結論。
- [SAGEO Arena, 2026 preprint](https://arxiv.org/abs/2602.12187)：指出 retrieval、rerank、generation 不同階段需要不同資訊；body-only 改寫若讓文件掉出 retrieval pool，可能反而降低可見度。
- [Temporal diversity in micro-video recommendation](https://link.springer.com/article/10.1007/s11063-024-11652-7)：支持長期興趣與短期興趣並存、避免連續同質推薦的研究方向；不是創作者平台保證。
- [GEO Citation Lab](https://github.com/yaojingang/geo-citation-lab)：可重現的 citation selection / absorption 分析工具與資料。
- [SafeGEO](https://github.com/QianfengWen/SafeGEO)：提醒 GEO 內容操縱可能把錯誤目標推高，因此本專案保留 source grounding，不把排名當成唯一目標。
- [GoogleChrome Lighthouse](https://github.com/GoogleChrome/lighthouse) 與 [web-vitals](https://github.com/GoogleChrome/web-vitals)：若日後把 SEO surface 擴到自有網站，可用來量測索引可達性與使用者體驗；目前沒有為社群 caption 增加依賴。
- [community geo-seo-aeo-skill](https://github.com/staksoft/geo-seo-aeo-skill) 與 [aeo-platform](https://github.com/webappski/aeo-platform)：可作為外部 audit 參考，但其規則與平台 visibility 宣稱未視為 Google 或社群平台官方訊號，也沒有直接加入 runtime dependency。

所有抓取到的研究快照位於 `docs/research/seo_aeo_2026/sources/`，方便之後重新驗證；官方頁面、平台政策與模型權重都可能更新。
