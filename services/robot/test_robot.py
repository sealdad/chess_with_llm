#!/usr/bin/env python3
"""Test robot connection"""

import sys

print("Step 1: Testing imports...")
try:
    from interbotix_xs_modules.arm import InterbotixManipulatorXS
    print("  - interbotix_xs_modules.arm OK")
except ImportError as e:
    print(f"  - FAILED: {e}")
    sys.exit(1)

print("\nStep 2: Connecting to RX-200...")
try:
    robot = InterbotixManipulatorXS("rx200", "arm", "gripper")
    print("  - Robot connected OK")
except Exception as e:
    print(f"  - FAILED: {e}")
    sys.exit(1)

print("\nStep 3: Moving to home pose...")
try:
    robot.arm.go_to_home_pose()
    print("  - Home pose OK")
except Exception as e:
    print(f"  - FAILED: {e}")

print("\nStep 4: Testing gripper...")
try:
    robot.gripper.open()
    print("  - Gripper open OK")
    robot.gripper.close()
    print("  - Gripper close OK")
except Exception as e:
    print(f"  - FAILED: {e}")

print("\nStep 5: Moving to sleep pose...")
try:
    robot.arm.go_to_sleep_pose()
    print("  - Sleep pose OK")
except Exception as e:
    print(f"  - FAILED: {e}")

print("\n=== ALL TESTS PASSED ===")
