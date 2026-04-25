#!/usr/bin/env python3
"""
Test script for chess vision pipeline - works offline with static images.

Usage:
    python test_vision_pipeline.py                    # Test with default image
    python test_vision_pipeline.py path/to/image.jpg  # Test with custom image
    python test_vision_pipeline.py --all              # Test all captured images
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from chess_vision.vision_pipeline import ChessVisionPipeline, analyze_image_file


def test_single_image(image_path: str, output_dir: str = "test_output"):
    """Test pipeline on a single image."""
    Path(output_dir).mkdir(exist_ok=True)

    image_name = Path(image_path).stem
    output_path = f"{output_dir}/{image_name}_analyzed.jpg"

    print(f"\n{'=' * 60}")
    print(f"Testing: {image_path}")
    print('=' * 60)

    result = analyze_image_file(
        image_path=image_path,
        output_path=output_path,
        print_result=True,
    )

    if result:
        # Also save the LLM prompt to a text file
        pipeline = ChessVisionPipeline(lazy_load=True)
        pipeline._det_model = True  # Skip loading for prompt generation

        llm_prompt = f"""The current chess board position is as follows:

FEN: {result.get_fen()}

Board visualization:
{result.get_ascii_board()}

{result.get_llm_description()}
"""
        prompt_path = f"{output_dir}/{image_name}_llm_prompt.txt"
        with open(prompt_path, "w") as f:
            f.write(llm_prompt)
        print(f"\n[Saved] LLM prompt to {prompt_path}")

    return result


def test_all_captures(captures_dir: str = "chess_vision/captures"):
    """Test pipeline on all captured images."""
    import glob

    images = glob.glob(f"{captures_dir}/*_color.jpg")
    if not images:
        print(f"No images found in {captures_dir}")
        return

    print(f"Found {len(images)} images to test")

    pipeline = ChessVisionPipeline()
    results = []

    for img_path in sorted(images)[:5]:  # Test first 5 to save time
        print(f"\n{'=' * 60}")
        print(f"Testing: {img_path}")

        result = pipeline.analyze_image(img_path)
        if result:
            print(f"FEN: {result.get_fen()}")
            print(f"Pieces detected: {len(result.board_state)}")
            print(f"Valid: {result.analysis['is_valid']}")
            results.append((img_path, result))
        else:
            print("Failed to analyze")
            results.append((img_path, None))

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print('=' * 60)
    successful = sum(1 for _, r in results if r is not None)
    print(f"Successful: {successful}/{len(results)}")


def interactive_test():
    """Interactive testing mode."""
    print("\n=== Chess Vision Pipeline - Interactive Test ===\n")

    pipeline = ChessVisionPipeline()

    while True:
        print("\nOptions:")
        print("  1. Analyze test.jpg")
        print("  2. Analyze custom image path")
        print("  3. Test with captured images")
        print("  4. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            test_single_image("test.jpg")
        elif choice == "2":
            path = input("Enter image path: ").strip()
            if Path(path).exists():
                test_single_image(path)
            else:
                print(f"File not found: {path}")
        elif choice == "3":
            test_all_captures()
        elif choice == "4":
            break
        else:
            print("Invalid option")


def main():
    parser = argparse.ArgumentParser(description="Test chess vision pipeline")
    parser.add_argument("image", nargs="?", default="test.jpg",
                        help="Image path to analyze (default: test.jpg)")
    parser.add_argument("--all", action="store_true",
                        help="Test all captured images")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--output", "-o", default="test_output",
                        help="Output directory")

    args = parser.parse_args()

    if args.interactive:
        interactive_test()
    elif args.all:
        test_all_captures()
    else:
        test_single_image(args.image, args.output)


if __name__ == "__main__":
    main()
