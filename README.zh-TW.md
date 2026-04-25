[English Version](README.md)

# Chess with LLM

**一個 LLM 為核心的西洋棋 AI Agent 框架—— 三種遊戲模式、具備個性與對話功能、預留視覺與機械手臂接口**

<!-- HERO GIF: 實體手臂抓子招牌鏡頭(3–6 秒,寬 720px)
<p align="center">
  <img src="docs/media/robot_hero.gif" alt="RX-200 手臂抓起棋子 demo" width="720">
</p>
-->

---

## 三種模式

| 模式 | 誰在下棋 | 對話 | 物理模式 |
|------|----------|----------|------|
| **Battle** | 你 vs AI(Stockfish 或 LLM 想旗路) | LLM 依照性格對旗步發表評論 | 可選:實體手臂或純模擬 |
| **Teach** | 你跟著課程走 | LLM 教練三階段講解 | 可選:實體手臂或純模擬 |
| **Watch** | LLM vs LLM | LLM 依照性格對旗步發表評論 | 純模擬 |


---

## Battle Mode — LLM 主導 × 實體手臂

<!-- SCREEN GIF: Battle 模式 UI(選角色 → 下 3 手 → AI trash talk 字幕,6–10 秒,寬 720px)
<p align="center">
  <img src="docs/media/screen_battle.gif" alt="Battle 模式螢幕錄影" width="720">
</p>
-->

***AI 怎麼決定走哪步***

Battle 模式的 `think` 節點支援兩種棋路決策方式：

```text
FEN + move history
       │
       ▼
  棋路決策模式
       │
       ├─ Stockfish 決定棋路
       │     └─ Stockfish 根據目前盤面產生 suggested_move
       │
       └─ LLM 決定棋路
             └─ LLM 根據盤面、歷史走法、角色與難度設定產生 suggested_move
       │
       ▼
  suggested_move (UCI)
       │
       ▼
  LLM 發表走棋評論
```

AI Agent 下棋時，核心棋路可以由 Stockfish 決定，LLM 僅負責根據角色發表評論；也可以切換成由 LLM 直接產生棋路，讓角色性格與難度更深度影響每一步決策。

UI 上可直接選擇 Stockfish / LLM1 / LLM2 / LLM3 作為對手引擎。其中 LLM1 ~ LLM3 是預設的三組 LLM provider / model 接口設定，可對應不同模型、角色或下棋風格。

每一步棋完成後，LLM 都會發表一句簡短評論。評論的語氣與內容會依照目前設定的 性格 決定。

***角色 × 評論 × 語音***

services/agent/conversation/modes.py 會將自由文字角色描述，例如「嘴砲海盜」、「禪修大師」、「街頭饒舌歌手」，透過關鍵字匹配到不同的情緒原型，再對應到 CosyVoice 的情緒 preset。(也可以接其他語音 api, 推薦 cosyvoice 本地模型可以有低延遲表現)

這讓 AI 對手不只會下棋，也會用符合人設的語氣評論每一步棋，並透過 TTS 呈現一致的角色聲音表現。

| 遊戲事件 | 嘴砲海盜                    | 禪修大師             |
| ---- | ----------------------- | ---------------- |
| 將軍   | `angry`，挑釁、壓迫感強         | `serious`，冷靜提醒局勢 |
| 將殺   | `angry + happy`，狂笑式勝利宣言 | `calm`，淡然結束對局    |
| 吃子   | `happy`，得意嘲諷            | `gentle`，平靜說明交換  |
| 一般走棋 | `disgusted`，嘴砲碎念        | `calm`，穩定分析      |


***實體視覺手臂接口***

這個 AI Agent 架構後續可以串接 **視覺系統** 與 **機械手臂**，讓下棋不只停留在螢幕上，而是可以實際操作棋盤上的棋子。

整體流程如下：


```text
人類走棋
   │
   ▼
RealSense 視覺偵測棋盤變化
   │
   ▼
判斷人類走了哪一步
   │
   ▼
AI Agent / LLM 決定回應棋路
   │
   ▼
將棋路轉成 UCI move
   │
   ▼
Robot service 轉成機械手臂動作
   │
   ▼
Interbotix RX200 執行 pick / place / capture
```

<!-- ROBOT GIF: Battle 模式實機 pick → move → place 一手(6–10 秒,寬 720px,可以分割畫面同步秀 UI)
<p align="center">
  <img src="docs/media/robot_battle.gif" alt="Battle 模式實機走棋" width="720">
</p>
-->

> 視覺 6 階段管線、AprilTag 手眼校準、64 格子教學、IK 搜尋策略、吃子/升變處理、YAML 設定檔參考,全部在 **[docs/ROBOT.zh-TW.md](docs/ROBOT.zh-TW.md)**。

---
## Teach Mode — AI 自動產生課程內容

Teach Mode 的核心優勢在於：**教學主題與課程大綱可以由 AI 制定**，AI 會根據教學目標自動發展成一連串步驟。

每個步驟都包含一個明確的教學重點，AI 會先說明概念，再示範棋步，最後引導學員自己操作。

<!-- SCREEN GIF: Teach 模式三階段 illustrate → demo → ending → explanation，約 8–12 秒，寬 720px
<p align="center">
  <img src="docs/media/screen_teach.gif" alt="Teach 模式螢幕錄影" width="720">
</p>
-->

課程內容使用 YAML 檔管理，例如 `lessons/*.yaml`。  
每一步可以包含固定的三階段敘事：

```yaml
lesson_id: 01_meet_the_pieces
title_zh: "認識棋子"
difficulty: beginner
steps:
  - fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    expected_move: "e2e4"
    instruction: "教學生認識兵的基本走法，並引導他將 e2 的兵走到 e4。"
    illustrate_zh: "兵是棋盤上最小但最勇敢的棋子。一次往前走一格，但第一步可以走兩格。"
    demo_narration_zh: "看，我指的是 e2 這顆兵。它現在可以往前走兩格，到達 e4。"
    ending_zh: "換你試試看，把 e2 的兵往前推到 e4。"
    hints_zh: ["點擊 e2 的兵，然後點擊 e4"]
    explanation_zh: "做得好！你的兵佔領了中心。兵向前走，但吃子時是斜著吃。"
```

教學流程如下：

```text
illustrate              demo                    ending
─────────────          ─────────────────       ──────────────────
AI 說明教學概念   ──▶   AI / 手臂示範棋步   ──▶   引導學生自己操作
(介紹本步重點)          (指出或示範走法)        學生下棋 → 系統驗證
                                                    │
                                                    ├─ 正確 → AI 稱讚 + 補充說明 + 進入下一步
                                                    └─ 錯誤 → AI 溫和糾正 + 提示 + 重新嘗試
```

每個步驟的目標不是單純「叫學生走一手棋」，而是讓 AI 針對該步驟建立完整的教學互動：

1. **illustrate**：先解釋這一步要學的觀念  
2. **demo**：示範正確棋步，必要時可搭配機械手臂指示棋盤位置  
3. **ending**：請學生實際操作  
4. **validation**：檢查學生是否走對  
5. **explanation**：根據結果給予稱讚、提示或補充說明  

---

### AI 自動補完課程內容

YAML 不一定要完整填寫所有敘事欄位。

如果 `illustrate_zh`、`demo_narration_zh`、`ending_zh`、`explanation_zh` 沒有事先寫好，LLM 會根據 `instruction`、目前 FEN、目標棋步與課程難度，即時生成對應的教學內容。

也就是說，建立課程時可以只提供最小資訊：

```yaml
steps:
  - fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    expected_move: "e2e4"
    instruction: "教學生認識兵的第一步可以走兩格，並引導他將 e2 的兵走到 e4。"
```

AI 會自動補出：

- 概念說明
- 示範旁白
- 學生操作指令
- 提示內容
- 完成後的補充解釋

這讓課程設計者不需要逐字撰寫所有教學台詞，只要定義 **教學意圖**，AI 就能發展成完整的互動式課程。

---

## Watch Mode — AI vs AI

Watch Mode 是讓不同 LLM 彼此下棋的觀察模式。

我知道 **LLM 不可能比得上 Stockfish 這類專用下棋引擎**。但我仍然很好奇：通用語言模型在沒有專門為西洋棋設計的情況下，到底能下到什麼程度？

因此做了 Watch Mode，讓不同 LLM 可以互相對戰，觀察它們在棋局理解、戰術判斷、犯錯頻率與風格表現上的差異。

<!-- SCREEN GIF: Watch 模式 LLM vs LLM，兩邊 chat label + 走棋 highlight，約 8–12 秒，寬 720px
<p align="center">
  <img src="docs/media/screen_watch.gif" alt="Watch 模式螢幕錄影" width="720">
</p>
-->

基本流程如下：

```text
白方：LLM1 + 角色設定
黑方：LLM2 / LLM3 + 角色設定

loop:
  讀取目前棋盤狀態
  判斷輪到哪一方
  呼叫該方 LLM 產生下一步棋
  執行走棋
  該方 LLM 依照角色發表評論
  更新棋盤狀態與勝負判定
```

每一方都可以指定不同的 LLM provider、模型與角色。  
例如：

```text
白方：LLM1，禪修大師
黑方：LLM3，街頭饒舌歌手
```

當白方下棋時，會使用白方的模型與角色來產生棋步和評論；  
當黑方下棋時，則使用黑方自己的模型與角色。

這不是由同一個旁白在解說兩邊，而是讓兩個 AI 對手各自用自己的「性格」下棋與說話。

---

### LLM 對戰觀察

我也使用 Watch Mode 測試了主流三大模型商的旗艦模型互相對戰, 難度都設定為 master, 性格都是職業棋手


**gpt 5.2 vs opus**

| # | white | black | winner |
|---|---|---|---|
| 1 | gpt5.5 | claude-opus-4-7 | claude-opus-4-7 |
| 2 | claude-opus-4-7| gpt 5.5 | draw |
| 3 | gpt 5.5 | claude-opus-4-7 | draw |
| 4 | claude-opus-4-7 | gpt 5.5 | claude-opus-4-7 

**gpt 5.5 vs gemini 3.1**

| # | white | black | winner |
|---|---|---|---|
| 1 | gpt5.5 | gemini 3.1 pro | gemini 3.1 pro |
| 2 | gemini 3.1 pro | gpt 5.5 | gemini 3.1 pro |
| 3 | gpt 5.2 | gemini 2.5 pro | draw |
| 4 | gemini 2.5 pro | gpt 5.2 | gemini 3.1 pro |

**claude opus 4.7 vs gemini 3.1 pro**

| # | white | black | winner |
|---|---|---|---|
| 1 | claude opus 4.7 | gemini 3.1 pro | gemini 3.1 pro |
| 2 | gemini 3.1 pro | claude opus 4.7 | gemini 3.1 pro |
| 3 | claude opus 4.7 | gemini 3.1 pro | gemini 3.1 pro |
| 4 | gemini 3.1 pro | claude opus 4.7 | gemini 3.1 pro |
|

註：超過一百個回合, 在 end game 階段只是在互相追逐就算 draw
目前測試下來，**Gemini 在下棋能力上明顯強很多**。  

---

## 快速開始(模擬模式)

不需要實體硬體,任何電腦上都能跑。整個 LLM 設定流程在 Web UI 上完成,**不需要設定環境變數**。

### 1. Clone 並啟動服務

```bash
git clone https://github.com/sealdad/chess-with-llm.git
cd chess-with-llm
docker-compose -f docker-compose.dev.yaml up
```

### 2. 打開 API Settings 面板

瀏覽器開 `http://localhost:8000`,點右上角 **API Settings**。

裡面有三組 LLM 欄位(LLM 1 / LLM 2 / LLM 3),只要填其中一組就能玩。三組都填的話 Battle 跟 Watch 模式可以互相切換或互戰。

### 3. 三大 provider 推薦設定

| Slot | Model | Base URL | API Key 取得 |
|------|-------|----------|----------------|
| **OpenAI** | `gpt-5.5` | `https://api.openai.com/v1` | [platform.openai.com](https://platform.openai.com/api-keys) |
| **Gemini** | `gemini-3.1-pro-preview` | `https://generativelanguage.googleapis.com/v1beta/openai/` | [Google AI Studio](https://aistudio.google.com/apikey) |
| **Claude** | `claude-opus-4-7` | `https://api.anthropic.com/v1/` | [console.anthropic.com](https://console.anthropic.com/settings/keys) |

填完按 **Save**,設定會寫進 `agent_settings.yaml`,下次啟動自動載入。

### 4. 開始下棋

回到主頁,勾選 **Simulation Mode**,選 **Battle / Teach / Watch** 任一模式,挑個角色性格(嘴砲海盜、禪修大師⋯⋯隨便寫),按 **Start Game**。

> **想接實體手臂?** 完整硬體版需要 Interbotix RX-200 + RealSense D435,首次啟動還要做 AprilTag 手眼校準與 64 格子教學。實體版設定、校準流程、API 契約全部在 **[docs/ROBOT.zh-TW.md](docs/ROBOT.zh-TW.md)**。

---

## 打造你自己的機器人

整套 agent 不綁死在 RX-200 上。Vision 和 Robot 都只是 FastAPI service——想接 UR5、DOBOT、自組機械臂,只要實作幾個 REST endpoint:

```
Vision:  POST /capture/occupancy   → 佔位 + 顏色
Robot:   POST /manual_pick         {square, piece_type}
         POST /manual_place        {square, piece_type}
         POST /capture             {square, piece_type}
         POST /arm/{work,vision}
```

**完整 API 契約、請求/回應格式、參考實作細節** → [docs/ROBOT.zh-TW.md](docs/ROBOT.zh-TW.md)


---

## 系統架構說明

串接 **LLM、棋局引擎、視覺系統與機械手臂** 的 AI Agent 架構。


```text
Browser UI
   │
   │ WebSocket / REST
   ▼
Agent Server :8000
   │
   ▼
LangGraph State Machine
   │
   ├─ observe
   ├─ detect_change
   ├─ think
   ├─ act
   ├─ verify
   └─ voice_announce
   │
   ├──────────────► Stockfish
   │
   ├──────────────► LLM Service
   │                 OpenAI / Gemini / Claude
   │
   ├──────────────► Vision Service :8001
   │                 RealSense D435
   │
   └──────────────► Robot Service :8002
                     Interbotix RX-200 + ROS
```

這樣的設計讓本專案不只是棋類 UI，而是一個完整的 **AI Agent + Vision + Robot** 實體互動系統。

---

## Multi-Provider LLM 相容層

已對三大 provider 做過調整, 確保呼叫正確. 

`services/agent/conversation/llm_service.py` 的 `_detect_provider()` + `_build_optional_params()`:

| Provider | 參數名 | temperature | Streaming |
|----------|--------|-------------|-----------|
| OpenAI | `max_completion_tokens` | ✓ | ✓ |
| Claude | `max_tokens` | ✗(Opus 已棄用) | ✓ |
| Gemini | `max_tokens × 10`(thinking 吃 budget) | ✓ | 非 thinking 模型才支援 |
| OpenRouter | `max_tokens` | ✓ | ✓ |

環境變數配置三組 LLM:

```bash
# LLM1
OPENAI_API_KEY=sk-...
LLM_MODEL1=gpt-4o

# LLM2
LLM_API_KEY2=...
LLM_MODEL2=gemini-2.0-flash
LLM_BASE_URL2=https://generativelanguage.googleapis.com/openai/

# LLM3
LLM_API_KEY3=sk-ant-...
LLM_MODEL3=claude-sonnet-4
LLM_BASE_URL3=https://api.anthropic.com/openai/
```

UI 上的 API Settings 面板可以同時設定三組,存成 `agent_settings.yaml` 持久化。

---

## 語音 Stack

```
麥克風
  │
  ▼
VAD(偵測語音邊界,濾掉 "um" / "嗯" / Whisper 幻覺)
  │
  ▼
Whisper STT ──▶ IntentRouter 分類:
                  MOVE / MODE_SWITCH / LANGUAGE_SWITCH /
                  GAME_COMMAND / CONVERSATION / IGNORE
  │
  ▼
LangGraph 相對節點
  │
  ▼
LLM 回應(streaming)
  │
  ▼
TTS(OpenAI / CosyVoice,情緒 preset)── WebSocket 逐句串流 ──▶ 瀏覽器
```

CosyVoice 可以部署在遠端 GPU 機器上,Agent 透過 HTTP 呼叫——0.5B 模型夠輕量,但情緒表現勝過 OpenAI TTS。

---

## 專案結構

```
services/
  agent/                   # 大腦 :8000
    main.py                   FastAPI、遊戲流程、LLM service 切換
    conversation/
      llm_service.py            multi-provider 相容層
      modes.py                  角色 × 情緒 × mode prompts
      intent_router.py          STT → intent 分類
    audio/
      stt_service.py            Whisper
      tts_service.py            OpenAI TTS
      tts_cosyvoice.py          CosyVoice 客戶端
    websocket/                voice_handler,雙向串流
    static/                   原生 JS Web UI(game/vision/robot/agent tabs)
    lessons.py                YAML 課程載入
    board_setup.py            自動棋盤整理演算法

  vision/                  # 眼睛 :8001
    main.py                   FastAPI 封裝 chess_vision

  robot/                   # 手 :8002
    main.py                   ROS + Interbotix,pick/place/capture

chess_vision/              # 視覺函式庫(相機無關)
  vision_pipeline.py          6 階段管線協調器
  chess_algo.py               YOLO 偵測 + 顏色分類
  board_state.py              格子映射、FEN 產生
  camera.py                   BaseCamera ABC + RealSense 實作
  handeye_calibration.py      AprilTag 手眼校準

rx200_agent/               # LangGraph agent
  graph.py                    狀態機組裝
  state.py                    AgentState TypedDict
  nodes/
    observe.py                視覺快照
    detect_change.py          佔位變化偵測
    think.py                  LLM / Stockfish 選棋
    act.py                    送 UCI 給 Robot
    voice_announce.py         角色語音評論
    error_handler.py          錯誤路由
  board_tracker.py            顏色過濾走棋偵測

lessons/                   # YAML 課程
docker-compose.yaml        # 完整硬體
docker-compose.dev.yaml    # 純模擬
```

---

## 技術棧

| 層級 | 技術 |
|------|------|
| Agent | FastAPI、LangGraph、python-chess、Python 3.10 |
| LLM | OpenAI GPT-4/4o、Google Gemini、Anthropic Claude、Stockfish 16 |
| 視覺 | YOLOv8(分割 + 偵測)、OpenCV、Intel RealSense D4xx |
| 語音 | Whisper STT、OpenAI TTS、CosyVoice(情緒 preset) |
| 機器人 | ROS 1 Noetic、Interbotix SDK、Dynamixel |
| 校準 | AprilTag (pupil-apriltags)、最小二乘手眼擬合 |
| 前端 | 原生 JS、WebSocket、MJPEG |
| 部署 | Docker Compose、volume-mounted YAML configs |

---

## 授權

MIT 

