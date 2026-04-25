# rx200_agent/nodes/think.py
"""
Think node - LLM reasoning to decide the robot's move.
"""

from typing import Optional
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from ..state import AgentState
from ..tools.stockfish_tool import StockfishTool
from ..utils.move_parser import get_game_status, is_in_check
from ..config import LLM_MODEL, LLM_TEMPERATURE


# Mode-specific personality instructions for chess play
MODE_PLAY_STYLES = {
    "coach": """You are in COACH mode. Your goal is to help the human learn.
- Consider moves that teach important chess principles
- Prefer instructive positions over purely optimal play
- If there's a clear teaching moment, you might play into it
- Explain your reasoning in an educational way""",

    "friend": """You are in FRIEND mode. Keep it fun and casual.
- Play good chess but don't be ruthless
- Mix in some creative or interesting moves
- Keep the game enjoyable for both players
- Be encouraging in your reasoning""",

    "opponent": """You are in OPPONENT mode. Play to win.
- Always choose the objectively strongest move
- Follow Stockfish closely
- Be clinical and precise
- Focus purely on winning the game""",

    "mean_friend": """You are in MEAN FRIEND mode. Competitive with playful trash talk.
- Play strong moves but with flair
- Choose moves that put pressure on your opponent
- If you can win material, go for it dramatically
- Your reasoning should have some playful banter""",
}


# Global tool instances
_stockfish_tool = None
_llm = None


def get_stockfish_tool() -> StockfishTool:
    """Get or create Stockfish tool instance."""
    global _stockfish_tool
    if _stockfish_tool is None:
        _stockfish_tool = StockfishTool(lazy_init=True)
    return _stockfish_tool


def get_llm() -> ChatOpenAI:
    """Get or create LLM instance."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
        )
    return _llm


def think_move(state: AgentState) -> dict:
    """
    LLM agent decides the robot's next move.

    This node:
    1. Gets Stockfish's recommendation
    2. Builds context for the LLM
    3. LLM decides whether to follow Stockfish or play differently
    4. Returns the suggested move

    Returns:
        Updated state fields with suggested_move
    """
    game = state["game"]
    current_fen = game["current_fen"]
    robot_color = game["robot_color"]

    print(f"\n[Agent] Thinking about my move...")

    # Get Stockfish recommendation
    stockfish = get_stockfish_tool()
    sf_result = stockfish._run(current_fen)

    if not sf_result["success"]:
        return {
            "error_type": "stockfish",
            "error_message": sf_result.get("error", "Stockfish failed"),
            "current_phase": "error",
        }

    stockfish_move = sf_result["best_move"]
    stockfish_eval = sf_result["evaluation"]
    top_moves = sf_result.get("top_moves", [])

    print(f"[Stockfish] Best move: {stockfish_move} (eval: {stockfish_eval})")

    # Build LLM prompt
    llm = get_llm()

    # Get mode-specific personality
    agent_mode = state.get("agent_mode", "friend")
    mode_instructions = MODE_PLAY_STYLES.get(agent_mode, MODE_PLAY_STYLES["friend"])

    system_prompt = f"""You are a chess-playing AI controlling a robot arm. You are playing as {robot_color}.

{mode_instructions}

Your task is to decide the best move for the current position. You have access to Stockfish's recommendation,
but you can choose to play differently based on your current mode and personality.

Respond with a JSON object containing:
- "move": the move in UCI format (e.g., "e2e4")
- "reasoning": brief explanation of your choice (in your personality style)
- "follow_stockfish": true/false whether you're following Stockfish's suggestion
"""

    # Build context about the game
    ascii_board = state.get("current_board", {}).get("ascii_board", "")
    last_human_move = game.get("last_human_move")

    human_move_str = ""
    if last_human_move:
        human_move_str = f"Human's last move: {last_human_move['piece']} {last_human_move['from_square']} -> {last_human_move['to_square']}"

    user_prompt = f"""Current position (FEN): {current_fen}

{ascii_board}

{human_move_str}

Stockfish analysis:
- Best move: {stockfish_move}
- Evaluation: {stockfish_eval}
- Top moves: {json.dumps(top_moves[:3]) if top_moves else 'N/A'}

Game status: Move {game['move_number']}, {'Check!' if game.get('is_check') else 'Not in check'}

What move should I play?
"""

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)
        response_text = response.content

        # Parse LLM response
        try:
            # Try to extract JSON from response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0]
            else:
                json_str = response_text

            decision = json.loads(json_str.strip())
            suggested_move = decision.get("move", stockfish_move)
            reasoning = decision.get("reasoning", "Following Stockfish")

        except (json.JSONDecodeError, IndexError):
            # If parsing fails, use Stockfish's move
            suggested_move = stockfish_move
            reasoning = f"LLM response parsing failed, using Stockfish: {stockfish_move}"

        print(f"[Agent] Decided to play: {suggested_move}")
        print(f"[Agent] Reasoning: {reasoning}")

        return {
            "suggested_move": suggested_move,
            "stockfish_best_move": stockfish_move,
            "stockfish_evaluation": stockfish_eval,
            "llm_reasoning": reasoning,
            "current_phase": "think",
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response_text},
            ],
        }

    except Exception as e:
        # LLM failed, fall back to Stockfish
        print(f"[Warning] LLM failed: {e}, using Stockfish move")

        return {
            "suggested_move": stockfish_move,
            "stockfish_best_move": stockfish_move,
            "stockfish_evaluation": stockfish_eval,
            "llm_reasoning": f"LLM error, using Stockfish: {stockfish_move}",
            "current_phase": "think",
        }


def think_move_stockfish_only(state: AgentState) -> dict:
    """
    Simpler version that just uses Stockfish without LLM reasoning.

    Useful when LLM is not available or for faster play.
    """
    game = state["game"]
    current_fen = game["current_fen"]

    print(f"\n[Agent] Calculating move...")

    stockfish = get_stockfish_tool()
    result = stockfish._run(current_fen)

    if not result["success"]:
        return {
            "error_type": "stockfish",
            "error_message": result.get("error", "Stockfish failed"),
            "current_phase": "error",
        }

    suggested_move = result["best_move"]
    evaluation = result["evaluation"]

    print(f"[Agent] Playing: {suggested_move} (eval: {evaluation})")

    return {
        "suggested_move": suggested_move,
        "stockfish_best_move": suggested_move,
        "stockfish_evaluation": evaluation,
        "llm_reasoning": f"Stockfish recommends {suggested_move}",
        "current_phase": "think",
    }
