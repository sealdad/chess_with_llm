#!/usr/bin/env python3
"""
Main entry point for the RX200 Chess Robot Agent.

Usage:
    python run_chess_agent.py                      # Robot plays black, uses LLM
    python run_chess_agent.py --robot-color white  # Robot plays white
    python run_chess_agent.py --manual             # Use manual trigger for human moves
    python run_chess_agent.py --no-llm             # Use only Stockfish (no LLM)
    python run_chess_agent.py --mock               # Use mock tools (no hardware)
    python run_chess_agent.py --debug              # Enable debug logging
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from rx200_agent.graph import create_chess_agent_graph
from rx200_agent.state import create_initial_state


def setup_logging(debug: bool = False):
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(name)s: %(message)s",
    )
    # Quiet down noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def run_agent(
    robot_color: str = "black",
    use_manual_trigger: bool = False,
    use_llm: bool = True,
    use_mock: bool = False,
    debug: bool = False,
):
    """
    Run the chess agent.

    Args:
        robot_color: "white" or "black"
        use_manual_trigger: If True, wait for ENTER instead of polling vision
        use_llm: If True, use LLM for move decisions
        use_mock: If True, use mock tools (no hardware)
        debug: If True, enable debug logging
    """
    setup_logging(debug)

    # Set up mock tools if requested
    if use_mock:
        print("[Mode] Using mock tools (no hardware)")
        # Patch the tool imports to use mock versions
        import rx200_agent.tools.vision_tool as vision_module
        import rx200_agent.tools.stockfish_tool as stockfish_module
        import rx200_agent.tools.robot_tool as robot_module

        vision_module._vision_tool = vision_module.MockVisionTool()
        stockfish_module._stockfish_tool = stockfish_module.MockStockfishTool()
        robot_module._robot_tool = robot_module.MockRobotTool()

        # Also patch the node modules
        import rx200_agent.nodes.observe as observe_module
        import rx200_agent.nodes.detect_change as detect_module
        import rx200_agent.nodes.think as think_module
        import rx200_agent.nodes.act as act_module

        observe_module._vision_tool = vision_module.MockVisionTool()
        detect_module._vision_tool = vision_module.MockVisionTool()
        think_module._stockfish_tool = stockfish_module.MockStockfishTool()
        act_module._robot_tool = robot_module.MockRobotTool()

    # Create the agent graph
    agent = create_chess_agent_graph(
        use_manual_trigger=use_manual_trigger,
        use_llm=use_llm,
    )

    # Create initial state
    initial_state = create_initial_state(robot_color)

    print("\n" + "=" * 60)
    print("  Starting Chess Game")
    print("=" * 60)
    print(f"  Mode: {'Mock' if use_mock else 'Hardware'}")
    print(f"  Move detection: {'Manual (ENTER)' if use_manual_trigger else 'Auto (polling)'}")
    print(f"  Move decision: {'LLM + Stockfish' if use_llm else 'Stockfish only'}")
    print("=" * 60 + "\n")

    try:
        # Run the agent graph
        final_state = None
        for state in agent.stream(initial_state):
            final_state = state

            # Check for end conditions
            game = state.get("game", {})
            if game.get("game_status") != "playing":
                break

            if not state.get("should_continue", True):
                break

        # Print final summary
        if final_state:
            print("\n" + "=" * 60)
            print("  Game Summary")
            print("=" * 60)

            game = final_state.get("game", {})
            print(f"  Status: {game.get('game_status', 'unknown')}")
            print(f"  Moves played: {game.get('move_number', 0) - 1}")
            print(f"  Final FEN: {game.get('current_fen', '')}")

            # Show last moves
            if game.get("last_human_move"):
                hm = game["last_human_move"]
                print(f"  Last human move: {hm['from_square']} -> {hm['to_square']}")

            if game.get("last_robot_move"):
                rm = game["last_robot_move"]
                print(f"  Last robot move: {rm['from_square']} -> {rm['to_square']}")

    except KeyboardInterrupt:
        print("\n\n[Interrupted] Game stopped by user.")

    print("\nThank you for playing!")


def main():
    parser = argparse.ArgumentParser(
        description="RX200 Chess Robot Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_chess_agent.py                     # Default: robot plays black, uses LLM
  python run_chess_agent.py --robot-color white # Robot plays white
  python run_chess_agent.py --manual            # Press ENTER to confirm human moves
  python run_chess_agent.py --no-llm            # Use only Stockfish for moves
  python run_chess_agent.py --mock              # Test without hardware
  python run_chess_agent.py --mock --manual     # Full mock testing mode
        """,
    )

    parser.add_argument(
        "--robot-color",
        choices=["white", "black"],
        default="black",
        help="Color the robot plays (default: black)",
    )

    parser.add_argument(
        "--manual",
        action="store_true",
        help="Use manual trigger (ENTER key) for human move detection",
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use only Stockfish for move decisions (no LLM)",
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock tools for testing without hardware",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    run_agent(
        robot_color=args.robot_color,
        use_manual_trigger=args.manual,
        use_llm=not args.no_llm,
        use_mock=args.mock,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
