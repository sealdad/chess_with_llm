#!/usr/bin/env python3
"""
Generate AprilTag PDF for printing.

Usage:
    python generate_apriltag_pdf.py --tag-size 60 --output apriltags.pdf
"""

import argparse
import numpy as np
from pathlib import Path

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
except ImportError:
    print("Please install reportlab: pip install reportlab")
    exit(1)

try:
    from pupil_apriltags import Detector
    HAS_PUPIL = True
except ImportError:
    HAS_PUPIL = False


# Official tag36h11 codes (36-bit values for the 6x6 inner data region).
# Extracted from the AprilTag C library via tag36h11_create().
# Full family has 587 codes; we include the first 30 for common use.
TAG36H11_CODES = {
    0: 0xd5d628584,  1: 0xd97f18b49,  2: 0xdd280910e,  3: 0xe479e9c98,
    4: 0xebcbca822,  5: 0xf31dab3ac,
}


def _load_all_codes():
    """Try to load all 587 tag36h11 codes from the C library at runtime."""
    try:
        import ctypes
        import apriltag
        det = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11'))
        libc = ctypes.CDLL(det.libc._name)
        libc.tag36h11_create.restype = ctypes.POINTER(apriltag._ApriltagFamily)
        fam = libc.tag36h11_create().contents
        codes = {}
        for i in range(fam.ncodes):
            codes[i] = fam.codes[i]
        return codes
    except Exception:
        return None


def generate_tag36h11(tag_id: int) -> np.ndarray:
    """
    Generate tag36h11 pattern for given ID.
    Returns 10x10 numpy array (1=white, 0=black).

    Layout: 10x10 grid
      - Row/col 0 and 9: white outer border
      - Row/col 1 and 8: black inner border
      - Rows 2-7, cols 2-7: 6x6 data bits from the 36-bit code
    """
    global TAG36H11_CODES

    if tag_id not in TAG36H11_CODES:
        # Try to load full code table from C library
        all_codes = _load_all_codes()
        if all_codes:
            TAG36H11_CODES = all_codes

    if tag_id not in TAG36H11_CODES:
        raise ValueError(
            f"Tag ID {tag_id} not available. Install 'apriltag' package "
            f"for IDs beyond {max(TAG36H11_CODES.keys())}: pip install apriltag"
        )

    code = TAG36H11_CODES[tag_id]

    # Build 10x10 pattern
    pattern = np.zeros((10, 10), dtype=np.uint8)

    # White outer border (row/col 0 and 9)
    pattern[0, :] = 1
    pattern[9, :] = 1
    pattern[:, 0] = 1
    pattern[:, 9] = 1

    # Inner border (row/col 1 and 8) stays black (0) — already zero

    # 6x6 data region from the 36-bit code (MSB first)
    for i in range(36):
        row = 2 + (i // 6)
        col = 2 + (i % 6)
        pattern[row, col] = (code >> (35 - i)) & 1

    return pattern


def draw_apriltag(c: canvas.Canvas, x: float, y: float,
                  tag_size_mm: float, tag_id: int):
    """
    Draw an AprilTag on the canvas.

    Args:
        c: ReportLab canvas
        x, y: Bottom-left position in points
        tag_size_mm: Size of the black square boundary in mm (8 inner cells)
        tag_id: Tag ID to generate
    """
    # tag36h11 is 10x10 grid: 1-cell white border + 8x8 black boundary
    # tag_size_mm refers to the black boundary (8 cells), not total (10 cells)
    cell_size = (tag_size_mm * mm) / 8
    tag_size = cell_size * 10  # Total drawn size including white border

    pattern = generate_tag36h11(tag_id)

    for row in range(10):
        for col in range(10):
            cell_x = x + col * cell_size
            cell_y = y + (9 - row) * cell_size  # Flip Y

            if pattern[row, col] == 1:
                c.setFillColorRGB(1, 1, 1)  # White
            else:
                c.setFillColorRGB(0, 0, 0)  # Black

            c.rect(cell_x, cell_y, cell_size, cell_size, fill=1, stroke=0)

    # Draw outer border for cutting guide
    c.setStrokeColorRGB(0.7, 0.7, 0.7)
    c.setLineWidth(0.5)
    c.rect(x, y, tag_size, tag_size, fill=0, stroke=1)


def create_apriltag_pdf(
    output_path: str,
    tag_size_mm: float = 60,
    tag_ids: list = None,
    include_info: bool = True,
):
    """
    Create PDF with AprilTags for printing.

    Args:
        output_path: Output PDF path
        tag_size_mm: Tag size in millimeters
        tag_ids: List of tag IDs to generate (default: [0, 1, 2, 3])
        include_info: Include size and ID information
    """
    if tag_ids is None:
        tag_ids = [0, 1, 2, 3, 4, 5]

    c = canvas.Canvas(output_path, pagesize=A4)
    page_width, page_height = A4

    margin = 20 * mm
    # Total drawn size = 10/8 * black boundary size (includes 1-cell white border)
    tag_size = (tag_size_mm / 8) * 10 * mm
    spacing = 10 * mm

    # Calculate grid
    cols = int((page_width - 2 * margin + spacing) / (tag_size + spacing))
    rows = int((page_height - 2 * margin - 30 * mm + spacing) / (tag_size + spacing + 15 * mm))

    cols = max(1, cols)
    rows = max(1, rows)

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, page_height - margin, "AprilTag tag36h11 - For Hand-Eye Calibration")

    c.setFont("Helvetica", 10)
    c.drawString(margin, page_height - margin - 15,
                 f"Tag size (black border): {tag_size_mm}mm | Total (with white border): {tag_size_mm * 10 / 8:.1f}mm")

    # Configuration info
    c.setFont("Helvetica", 9)
    y_info = page_height - margin - 35
    c.drawString(margin, y_info, "Configuration for apriltag_detector.py:")
    c.setFont("Courier", 9)
    c.drawString(margin + 10, y_info - 12, f'tag_family = "tag36h11"')
    c.drawString(margin + 10, y_info - 24, f'tag_size = {tag_size_mm / 1000:.3f}  # {tag_size_mm}mm in meters')

    start_y = page_height - margin - 80 * mm

    tag_idx = 0
    while tag_idx < len(tag_ids):
        for row in range(rows):
            for col in range(cols):
                if tag_idx >= len(tag_ids):
                    break

                tag_id = tag_ids[tag_idx]
                x = margin + col * (tag_size + spacing)
                y = start_y - row * (tag_size + spacing + 15 * mm)

                if y < margin:
                    break

                draw_apriltag(c, x, y, tag_size_mm, tag_id)

                # Label
                if include_info:
                    c.setFont("Helvetica", 9)
                    c.setFillColorRGB(0, 0, 0)
                    total_mm = tag_size_mm * 10 / 8
                    c.drawString(x, y - 12, f"ID: {tag_id} | Black border: {tag_size_mm}mm | Total: {total_mm:.1f}mm")

                tag_idx += 1

        if tag_idx < len(tag_ids):
            c.showPage()
            start_y = page_height - margin - 20 * mm

    # Footer with instructions
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.3, 0.3, 0.3)
    footer_y = margin - 5
    c.drawString(margin, footer_y + 20, "Instructions:")
    c.drawString(margin, footer_y + 8, "1. Print at 100% scale (no scaling)")
    c.drawString(margin, footer_y - 4, "2. Verify size with ruler after printing")
    c.drawString(margin, footer_y - 16, "3. Mount on flat, rigid surface")

    c.save()
    print(f"Created: {output_path}")
    print(f"  - Tag family: tag36h11")
    print(f"  - Tag size (black border): {tag_size_mm}mm")
    print(f"  - Total size (with white border): {tag_size_mm * 10 / 8:.1f}mm")
    print(f"  - Tag IDs: {tag_ids[:tag_idx]}")


def main():
    parser = argparse.ArgumentParser(description="Generate AprilTag PDF for printing")
    parser.add_argument("--tag-size", type=float, default=60,
                        help="Tag size in mm (default: 60)")
    parser.add_argument("--tag-ids", type=str, default="0,1,2,3,4,5",
                        help="Comma-separated tag IDs (default: 0,1,2,3,4,5)")
    parser.add_argument("--output", "-o", type=str, default="apriltags_60mm.pdf",
                        help="Output PDF path")

    args = parser.parse_args()

    tag_ids = [int(x.strip()) for x in args.tag_ids.split(",")]

    create_apriltag_pdf(
        output_path=args.output,
        tag_size_mm=args.tag_size,
        tag_ids=tag_ids,
    )


if __name__ == "__main__":
    main()
