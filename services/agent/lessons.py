"""
Lesson loader and data models for Teach mode.

Lessons are YAML files in the /lessons directory with structured steps.
Each step has a FEN position, instruction, optional expected move, hints, and explanation.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict
from pydantic import BaseModel

# Traditional → Simplified Chinese converter
try:
    from opencc import OpenCC
    _t2s = OpenCC('t2s').convert
except ImportError:
    _t2s = lambda text: text  # fallback: return as-is


class LessonStep(BaseModel):
    """A single step in a lesson.

    Each step has 3 teaching phases (all optional, LLM generates from instruction if missing):
      1. illustrate  — explain the concept (no robot movement)
      2. demo_narration — narrate while robot hovers the move
      3. ending — short wrap-up, invite student to try
    """
    fen: str
    instruction: str
    instruction_zh: str = ""
    expected_move: Optional[str] = None
    hints: List[str] = []
    hints_zh: List[str] = []
    explanation: str = ""
    explanation_zh: str = ""
    # 3-phase teaching text (optional — LLM fills in from instruction if absent)
    illustrate: str = ""
    illustrate_zh: str = ""
    demo_narration: str = ""
    demo_narration_zh: str = ""
    ending: str = ""
    ending_zh: str = ""


class Lesson(BaseModel):
    """A complete lesson with metadata and steps."""
    lesson_id: str
    title: str
    title_zh: str = ""
    description: str
    description_zh: str = ""
    difficulty: str = "beginner"
    steps: List[LessonStep]


# Lesson directory: check /app/lessons (container) first, fall back to project root
_LESSONS_DIR = Path("/app/lessons") if Path("/app/lessons").exists() else Path(__file__).parent.parent.parent / "lessons"


def load_lesson(lesson_id: str) -> Optional[Lesson]:
    """Load a single lesson by ID from YAML file."""
    import yaml

    lesson_file = _LESSONS_DIR / f"{lesson_id}.yaml"
    if not lesson_file.exists():
        return None

    try:
        with open(lesson_file, "r") as f:
            data = yaml.safe_load(f)

        steps = []
        for step_data in data.get("steps", []):
            steps.append(LessonStep(
                fen=step_data["fen"],
                instruction=step_data.get("instruction", ""),
                instruction_zh=step_data.get("instruction_zh", ""),
                expected_move=step_data.get("expected_move"),
                hints=step_data.get("hints", []),
                hints_zh=step_data.get("hints_zh", []),
                explanation=step_data.get("explanation", ""),
                explanation_zh=step_data.get("explanation_zh", ""),
                illustrate=step_data.get("illustrate", ""),
                illustrate_zh=step_data.get("illustrate_zh", ""),
                demo_narration=step_data.get("demo_narration", ""),
                demo_narration_zh=step_data.get("demo_narration_zh", ""),
                ending=step_data.get("ending", ""),
                ending_zh=step_data.get("ending_zh", ""),
            ))

        return Lesson(
            lesson_id=data["lesson_id"],
            title=data["title"],
            title_zh=data.get("title_zh", ""),
            description=data["description"],
            description_zh=data.get("description_zh", ""),
            difficulty=data.get("difficulty", "beginner"),
            steps=steps,
        )
    except Exception as e:
        print(f"[Lessons] Failed to load {lesson_id}: {e}")
        return None


def list_lessons() -> List[Dict]:
    """List all available lessons with metadata (without loading full steps)."""
    import yaml

    lessons = []
    if not _LESSONS_DIR.exists():
        return lessons

    for f in sorted(_LESSONS_DIR.glob("*.yaml")):
        try:
            with open(f, "r") as fh:
                data = yaml.safe_load(fh)
            title_zh = data.get("title_zh", "")
            desc_zh = data.get("description_zh", "")
            lessons.append({
                "lesson_id": data["lesson_id"],
                "title": data["title"],
                "title_zh": title_zh,
                "title_zh_cn": _t2s(title_zh) if title_zh else "",
                "description": data["description"],
                "description_zh": desc_zh,
                "description_zh_cn": _t2s(desc_zh) if desc_zh else "",
                "difficulty": data.get("difficulty", "beginner"),
                "step_count": len(data.get("steps", [])),
            })
        except Exception as e:
            print(f"[Lessons] Failed to read {f.name}: {e}")

    return lessons
