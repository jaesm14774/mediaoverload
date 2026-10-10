# MediaOverload Agentic

MediaOverload 的 Python 執行套件，提供目標、執行計畫、技能／工具協調、媒體檢查與發布。設定與模型工作流由 repo checkout 的 `configs/` 提供。

```powershell
# 從 repo 根目錄
python -m pip install -e "./agentic[dev]"
agentic --goal "A hero catches a falling seed" --media-type image
```

安裝與操作請看 [專案 README](../README.md)。共用邊界、策略與模型擴充方式請看 [架構指南](../docs/architecture.md)。此文件不另維護一份策略或環境變數清單。
