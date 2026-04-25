"""
Game modes and personality configuration.

Two game modes: Battle and Teach.
Battle mode uses free-text character personality + configurable difficulty.
Teach mode uses structured lessons with guided practice.

Supports dual-language: English (en) and Traditional Chinese (zh-TW).
"""

from enum import Enum
from typing import Dict, Optional


class GameMode(Enum):
    """Available game modes."""
    BATTLE = "battle"
    TEACH = "teach"
    WATCH = "watch"


# Backward-compatible alias
AgentMode = GameMode

# Difficulty levels map to Stockfish depth
DIFFICULTY_DEPTH = {
    "beginner": 1,
    "intermediate": 5,
    "advanced": 12,
    "master": 18,
}

# Default TTS voice per game mode
MODE_VOICES: Dict[GameMode, str] = {
    GameMode.BATTLE: "echo",   # Energetic, competitive
    GameMode.TEACH: "nova",    # Warm, educational
}

# ── Character-based emotion profiles for CosyVoice ──
# Maps character archetype keywords → emotion overrides per game event.
# Events: checkmate, check_capture, check, capture, normal, welcome, goodbye.
# Default profile is used when no keywords match.
CHARACTER_EMOTION_PROFILES: Dict[str, Dict[str, str]] = {
    "_default": {
        "checkmate": "happy",
        "check_capture": "happy",
        "check": "serious",
        "capture": "happy",
        "normal": "calm",
        "welcome": "happy",
        "goodbye": "gentle",
    },
    "aggressive": {
        "checkmate": "angry",
        "check_capture": "angry",
        "check": "angry",
        "capture": "happy",
        "normal": "serious",
        "welcome": "serious",
        "goodbye": "serious",
    },
    "friendly": {
        "checkmate": "happy",
        "check_capture": "happy",
        "check": "surprised",
        "capture": "happy",
        "normal": "gentle",
        "welcome": "happy",
        "goodbye": "gentle",
    },
    "calm": {
        "checkmate": "calm",
        "check_capture": "calm",
        "check": "calm",
        "capture": "calm",
        "normal": "calm",
        "welcome": "calm",
        "goodbye": "calm",
    },
    "mean": {
        "checkmate": "happy",
        "check_capture": "happy",
        "check": "angry",
        "capture": "angry",
        "normal": "disgusted",
        "welcome": "angry",
        "goodbye": "disgusted",
    },
    "zen": {
        "checkmate": "gentle",
        "check_capture": "calm",
        "check": "calm",
        "capture": "calm",
        "normal": "gentle",
        "welcome": "gentle",
        "goodbye": "gentle",
    },
    "pirate": {
        "checkmate": "happy",
        "check_capture": "angry",
        "check": "angry",
        "capture": "happy",
        "normal": "serious",
        "welcome": "happy",
        "goodbye": "serious",
    },
    "sarcastic": {
        "checkmate": "surprised",
        "check_capture": "happy",
        "check": "surprised",
        "capture": "happy",
        "normal": "disgusted",
        "welcome": "disgusted",
        "goodbye": "calm",
    },
    "coach": {
        "checkmate": "encouraging",
        "check_capture": "encouraging",
        "check": "encouraging",
        "capture": "happy",
        "normal": "encouraging",
        "welcome": "encouraging",
        "goodbye": "encouraging",
    },
    "sad": {
        "checkmate": "sad",
        "check_capture": "sad",
        "check": "fearful",
        "capture": "sad",
        "normal": "sad",
        "welcome": "sad",
        "goodbye": "sad",
    },
}

# Keywords that map to each profile (checked against lowercased character string)
_PROFILE_KEYWORDS: Dict[str, list] = {
    "aggressive": ["aggressive", "tough", "fierce", "warrior", "fighter", "brutal",
                    "violent", "intense", "hardcore"],
    "friendly": ["friendly", "friend", "nice", "kind", "warm", "cheerful", "sweet",
                 "buddy", "pal", "朋友", "親切"],
    "calm": ["calm", "quiet", "chill", "relaxed", "stoic", "冷靜"],
    "mean": ["mean", "trash", "bully", "rude", "toxic", "cruel", "毒舌", "嘲諷"],
    "zen": ["zen", "master", "sage", "buddha", "monk", "wise", "philosopher",
            "禪", "大師", "智者"],
    "pirate": ["pirate", "captain", "海盜", "船長"],
    "sarcastic": ["sarcastic", "ironic", "cynical", "witty", "dry humor", "諷刺"],
    "coach": ["coach", "teacher", "mentor", "instructor", "教練", "老師"],
    "sad": ["sad", "depressed", "gloomy", "melancholy", "悲傷", "憂鬱"],
}


def resolve_character_emotion(character: str, event: str) -> str:
    """Resolve the CosyVoice emotion for a game event based on character personality.

    Args:
        character: Free-text character description (e.g. "trash-talking pirate").
        event: One of "checkmate", "check_capture", "check", "capture",
               "normal", "welcome", "goodbye".

    Returns:
        Emotion string matching CosyVoice EMOTION_PRESETS keys.
    """
    if not character:
        return CHARACTER_EMOTION_PROFILES["_default"].get(event, "calm")

    char_lower = character.lower()

    # Check each profile's keywords against the character string
    for profile_name, keywords in _PROFILE_KEYWORDS.items():
        for kw in keywords:
            if kw in char_lower:
                profile = CHARACTER_EMOTION_PROFILES[profile_name]
                return profile.get(event, CHARACTER_EMOTION_PROFILES["_default"].get(event, "calm"))

    return CHARACTER_EMOTION_PROFILES["_default"].get(event, "calm")

# Human-readable names
MODE_NAMES: Dict[str, Dict[GameMode, str]] = {
    "en": {
        GameMode.BATTLE: "Battle Mode",
        GameMode.TEACH: "Teach Mode",
    },
    "zh-TW": {
        GameMode.BATTLE: "對戰模式",
        GameMode.TEACH: "教學模式",
    },
}


def _is_zh_lang(language: str) -> bool:
    """Check if language is any Chinese variant."""
    return language in ("zh-TW", "zh-CN") or language.startswith("zh")


def build_battle_prompt(character: str, language: str = "en") -> str:
    """Build a system prompt for battle mode from a free-text character description.

    Args:
        character: Free-text personality description (e.g. "trash-talking pirate",
                   "zen master", "sarcastic teenager"). If empty, uses a default.
        language: Current language setting.

    Returns:
        Full system prompt string.
    """
    character = (character or "").strip()

    if _is_zh_lang(language):
        lang_note = "請用繁體中文回答。" if language == "zh-TW" else "请用简体中文回答。" if language == "zh-CN" else f"請用{language}回答。"
        if not character:
            character = "一個有競爭力又愛開玩笑的朋友"
        return (
            f"你是一個西洋棋對手。你的個性是：{character}。\n\n"
            f"規則：\n"
            f"- 完全沉浸在這個角色中\n"
            f"- 對棋步和局面做出符合角色的反應\n"
            f"- 回答只用1句話，要簡短有個性！\n"
            f"- {lang_note}"
        )
    elif language != "en":
        # Non-English, non-Chinese language
        if not character:
            character = "a competitive friend who loves playful trash talk"
        return (
            f"You are a chess opponent. Your personality: {character}.\n\n"
            f"Rules:\n"
            f"- Stay fully in character at all times\n"
            f"- React to moves and positions in your character's style\n"
            f"- Keep responses to 1 short sentence. Be brief and in character!\n"
            f"- IMPORTANT: Respond in {language}."
        )
    else:
        if not character:
            character = "a competitive friend who loves playful trash talk"
        return (
            f"You are a chess opponent. Your personality: {character}.\n\n"
            f"Rules:\n"
            f"- Stay fully in character at all times\n"
            f"- React to moves and positions in your character's style\n"
            f"- Keep responses to 1 short sentence. Be brief and in character!"
        )


def build_teach_prompt(language: str = "en", lesson_context: str = "") -> str:
    """Build a system prompt for teach mode.

    Args:
        language: Current language setting.
        lesson_context: Optional context about the current lesson step.

    Returns:
        Full system prompt string.
    """
    if _is_zh_lang(language):
        lang_note = "請用繁體中文回答。" if language == "zh-TW" else "请用简体中文回答。" if language == "zh-CN" else f"請用{language}回答。"
        base = (
            "你是一位友善且有耐心的西洋棋教練，正在指導學生完成一堂練習課。\n\n"
            "你的角色：\n"
            "- 用簡單、清楚的方式解釋概念\n"
            "- 鼓勵好的走法，溫和地糾正錯誤\n"
            "- 引導學生思考，而非直接告訴答案\n"
            "- 用2-3句話回答，簡潔但有教育意義\n"
            f"- {lang_note}"
        )
    elif language != "en":
        base = (
            "You are a friendly, patient chess coach guiding a student through a practice lesson.\n\n"
            "Your role:\n"
            "- Explain concepts in simple, clear terms\n"
            "- Encourage good play and gently correct mistakes\n"
            "- Guide the student to think, don't just give answers\n"
            "- Keep responses to 2-3 sentences, concise but educational\n"
            f"- IMPORTANT: Respond in {language}."
        )
    else:
        base = (
            "You are a friendly, patient chess coach guiding a student through a practice lesson.\n\n"
            "Your role:\n"
            "- Explain concepts in simple, clear terms\n"
            "- Encourage good play and gently correct mistakes\n"
            "- Guide the student to think, don't just give answers\n"
            "- Keep responses to 2-3 sentences, concise but educational"
        )

    if lesson_context:
        base += f"\n\n--- Current Lesson Context ---\n{lesson_context}"

    return base


# Legacy prompt lookup (for backward compatibility with code that calls get_mode_prompt)
MODE_PROMPTS: Dict[str, Dict[GameMode, str]] = {
    "en": {
        GameMode.BATTLE: build_battle_prompt("", "en"),
        GameMode.TEACH: build_teach_prompt("en"),
    },
    "zh-TW": {
        GameMode.BATTLE: build_battle_prompt("", "zh-TW"),
        GameMode.TEACH: build_teach_prompt("zh-TW"),
    },
}


# Mode greetings
MODE_GREETINGS: Dict[str, Dict[GameMode, str]] = {
    "en": {
        GameMode.BATTLE: "Battle mode! Let's see what you've got!",
        GameMode.TEACH: "Teach mode activated! Let's learn together. I'll guide you through the lesson.",
    },
    "zh-TW": {
        GameMode.BATTLE: "對戰模式！讓我看看你的實力！",
        GameMode.TEACH: "教學模式啟動！我們一起學習，我會引導你完成課程。",
    },
}

# Language switch trigger phrases
LANGUAGE_TRIGGERS: Dict[str, list] = {
    "en": [
        "switch to english", "english mode", "speak english",
        "english please", "change to english",
    ],
    "zh-TW": [
        "切換中文", "中文模式", "說中文", "請說中文", "換成中文",
        "switch to chinese", "chinese mode", "speak chinese",
        "traditional chinese", "mandarin mode",
    ],
}


def get_mode_name(mode: GameMode, language: str = "en") -> str:
    """Get human-readable mode name for a language."""
    lang_names = MODE_NAMES.get(language, MODE_NAMES["en"])
    return lang_names.get(mode, mode.value)


def get_mode_prompt(mode: GameMode, language: str = "en") -> str:
    """Get mode system prompt for a language."""
    lang_prompts = MODE_PROMPTS.get(language, MODE_PROMPTS["en"])
    return lang_prompts.get(mode, lang_prompts[GameMode.BATTLE])


def detect_mode_from_text(text: str, language: str = "en") -> Optional[GameMode]:
    """Detect if text contains a mode switch request."""
    text_lower = text.lower()

    battle_triggers = {
        "en": ["battle mode", "fight mode", "versus mode", "let's battle", "let's fight"],
        "zh-TW": ["對戰模式", "戰鬥模式", "來對戰", "比賽模式"],
    }
    teach_triggers = {
        "en": ["teach mode", "lesson mode", "learn mode", "teach me", "learning mode"],
        "zh-TW": ["教學模式", "學習模式", "教我", "上課模式"],
    }

    for lang_triggers in [battle_triggers.get(language, []), battle_triggers.get("en", [])]:
        for trigger in lang_triggers:
            if trigger in text_lower:
                return GameMode.BATTLE

    for lang_triggers in [teach_triggers.get(language, []), teach_triggers.get("en", [])]:
        for trigger in lang_triggers:
            if trigger in text_lower:
                return GameMode.TEACH

    return None


def detect_language_from_text(text: str) -> Optional[str]:
    """Detect if text contains a language switch request."""
    text_lower = text.lower()
    for lang, triggers in LANGUAGE_TRIGGERS.items():
        for trigger in triggers:
            if trigger in text_lower:
                return lang
    return None


def get_mode_greeting(mode: GameMode, language: str = "en") -> str:
    """Get a greeting message when switching to a mode."""
    lang_greetings = MODE_GREETINGS.get(language, MODE_GREETINGS["en"])
    return lang_greetings.get(mode, f"Switched to {get_mode_name(mode, language)}.")


def get_language_greeting(language: str) -> str:
    """Get a greeting message when switching language."""
    if language == "zh-TW":
        return "已切換到繁體中文模式。有什麼我可以幫你的嗎？"
    return "Switched to English mode. How can I help you?"
