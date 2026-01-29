#!/usr/bin/env python3
"""
Generate high-quality screenshots from the first frame of safety environments.

Usage:
    python scripts/generate_env_screenshots.py --output_dir results/plots/env_screenshots
    python scripts/generate_env_screenshots.py --tile_size 64 --output_dir results/plots/env_screenshots
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_PATH))

# Import environments
from env.safety.dynamic_lava_room import DynamicCrossing
from env.safety.fragile_carry import FragileCrossingEnv
from env.safety.corner import Corner


def capture_screenshot(env, tile_size: int = 32, highlight: bool = True) -> np.ndarray:
    """
    Capture a high-quality screenshot from an environment.
    
    Args:
        env: The MiniGrid environment
        tile_size: Size of each tile in pixels (higher = better quality)
        highlight: Whether to highlight visible cells
    
    Returns:
        RGB image as numpy array
    """
    # Get the full render (not POV)
    img = env.get_full_render(highlight=highlight, tile_size=tile_size)
    return img


def save_screenshot(img: np.ndarray, filepath: str, format: str = "png"):
    """Save the screenshot to a file."""
    pil_img = Image.fromarray(img)
    pil_img.save(filepath, format=format.upper())
    print(f"Saved: {filepath}")


def pad_to_square(img: np.ndarray, bg_color=(255, 255, 255)) -> np.ndarray:
    """Pad an image to make it square by adding whitespace."""
    h, w = img.shape[:2]
    if h >= w:
        return img
    # Need to add padding to top and bottom
    pad_total = w - h
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    
    # Create padding arrays
    top_pad = np.full((pad_top, w, 3), bg_color, dtype=np.uint8)
    bottom_pad = np.full((pad_bottom, w, 3), bg_color, dtype=np.uint8)
    
    return np.concatenate([top_pad, img, bottom_pad], axis=0)


def main():
    parser = argparse.ArgumentParser(description="Generate environment screenshots")
    parser.add_argument("--output_dir", type=str, default="results/plots/env_screenshots",
                        help="Output directory for screenshots")
    parser.add_argument("--tile_size", type=int, default=48,
                        help="Tile size in pixels (higher = better quality). Default: 48")
    parser.add_argument("--format", type=str, default="png", choices=["png", "jpg", "pdf"],
                        help="Image format")
    parser.add_argument("--no_highlight", action="store_true",
                        help="Disable cell highlighting")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Create output directory
    output_dir = ROOT_PATH / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    highlight = not args.no_highlight

    # Define environments to capture (name, class, kwargs, pad_to_square)
    environments = [
        ("dynamic_crossing", DynamicCrossing, {"size": 9, "render_mode": "rgb_array"}, False),
        ("fragile_crossing", FragileCrossingEnv, {"width": 15, "height": 7, "render_mode": "rgb_array"}, True),
        ("four_corner", Corner, {"width": 13, "height": 13, "render_mode": "rgb_array"}, False),
    ]

    print(f"Generating screenshots with tile_size={args.tile_size}...")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    for name, EnvClass, kwargs, do_pad in environments:
        try:
            # Create environment
            env = EnvClass(**kwargs)
            
            # Reset to get initial state
            env.reset(seed=args.seed)
            
            # Capture screenshot
            img = capture_screenshot(env, tile_size=args.tile_size, highlight=highlight)
            if do_pad:
                img = pad_to_square(img)
            
            # Save
            filename = f"{name}.{args.format}"
            filepath = output_dir / filename
            save_screenshot(img, str(filepath), format=args.format)
            
            # Also save a version without highlight for cleaner figures
            if highlight:
                img_no_highlight = capture_screenshot(env, tile_size=args.tile_size, highlight=False)
                if do_pad:
                    img_no_highlight = pad_to_square(img_no_highlight)
                filename_clean = f"{name}_clean.{args.format}"
                filepath_clean = output_dir / filename_clean
                save_screenshot(img_no_highlight, str(filepath_clean), format=args.format)
            
            env.close()
            
        except Exception as e:
            print(f"Error with {name}: {e}")
            import traceback
            traceback.print_exc()

    print("-" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
