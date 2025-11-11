#!/usr/bin/env python
"""
Helper script to set up cache configuration for BERT benchmarking.
Creates a .env file with appropriate cache directory settings.
"""

import os
import sys
import shutil
from pathlib import Path


def get_disk_usage():
    """Get disk usage for all available drives."""
    drives = []

    if sys.platform == "win32":
        # Windows: Check all drive letters
        import string
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                try:
                    usage = shutil.disk_usage(drive)
                    drives.append({
                        'path': drive,
                        'free_gb': usage.free / (1024**3),
                        'total_gb': usage.total / (1024**3)
                    })
                except:
                    pass
    else:
        # Linux/Mac: Check common mount points
        common_paths = [
            "/",
            "/home",
            "/mnt",
            "/media",
            str(Path.home())
        ]

        for path in common_paths:
            if os.path.exists(path):
                try:
                    usage = shutil.disk_usage(path)
                    drives.append({
                        'path': path,
                        'free_gb': usage.free / (1024**3),
                        'total_gb': usage.total / (1024**3)
                    })
                except:
                    pass

    return drives


def suggest_cache_location(drives, min_free_gb=20):
    """Suggest best cache location based on available space."""
    suitable = [d for d in drives if d['free_gb'] >= min_free_gb]

    if not suitable:
        return None

    # Prefer drives with most free space
    suitable.sort(key=lambda x: x['free_gb'], reverse=True)
    return suitable[0]


def create_env_file(cache_dir):
    """Create .env file with specified cache directory."""
    env_content = f"""# BERT Benchmarking - Cache Configuration
# Models will download to this location (can be 10-50GB total)

BERT_CACHE_DIR={cache_dir}

# Advanced settings (automatically derived from BERT_CACHE_DIR)
# Uncomment and modify if you want to override specific locations:
# HF_HOME={cache_dir}/huggingface
# TRANSFORMERS_CACHE={cache_dir}/transformers
# HF_DATASETS_CACHE={cache_dir}/datasets
# TORCH_HOME={cache_dir}/torch
"""

    with open(".env", "w") as f:
        f.write(env_content)

    print(f"✓ Created .env file with cache directory: {cache_dir}")


def main():
    print("="*60)
    print("BERT Benchmarking - Cache Setup")
    print("="*60)
    print()

    # Check if .env already exists
    if os.path.exists(".env"):
        print("⚠ .env file already exists!")
        response = input("Overwrite it? (y/n): ").strip().lower()
        if response != 'y':
            print("Setup cancelled.")
            return
        print()

    # Get available drives
    print("Scanning available drives...")
    drives = get_disk_usage()

    if not drives:
        print("❌ Could not detect any drives!")
        return

    print()
    print("Available storage locations:")
    print("-" * 60)
    for i, drive in enumerate(drives, 1):
        print(f"{i}. {drive['path']:<20} "
              f"Free: {drive['free_gb']:>8.1f} GB / "
              f"Total: {drive['total_gb']:>8.1f} GB")
    print()

    # Suggest best location
    suggested = suggest_cache_location(drives)
    if suggested:
        print(f"💡 Recommended: {suggested['path']} "
              f"({suggested['free_gb']:.1f} GB free)")
    else:
        print("⚠ Warning: All drives have less than 20GB free space")
        print("   BERT models may require 10-50GB total")

    print()
    print("Options:")
    print("  1-{}: Select a drive from the list above".format(len(drives)))
    print("  c: Enter custom path")
    print("  d: Use default (./cache in project directory)")
    print("  q: Quit without creating .env")
    print()

    choice = input("Your choice: ").strip().lower()

    if choice == 'q':
        print("Setup cancelled.")
        return
    elif choice == 'd':
        cache_dir = "./cache"
    elif choice == 'c':
        cache_dir = input("Enter cache directory path: ").strip()
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(drives):
                base_path = drives[idx]['path']

                # Suggest subdirectory
                if sys.platform == "win32":
                    cache_dir = os.path.join(base_path, "bert_cache")
                else:
                    cache_dir = os.path.join(base_path, ".cache", "bert")

                print(f"\nWill use: {cache_dir}")
                confirm = input("OK? (y/n): ").strip().lower()
                if confirm != 'y':
                    cache_dir = input("Enter custom path: ").strip()
            else:
                print("Invalid selection!")
                return
        except ValueError:
            print("Invalid input!")
            return

    # Create .env file
    print()
    create_env_file(cache_dir)

    # Create cache directory
    try:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"✓ Created cache directory: {cache_dir}")
    except Exception as e:
        print(f"⚠ Could not create directory: {e}")
        print("  You may need to create it manually or use a different location")

    print()
    print("="*60)
    print("Setup complete!")
    print("="*60)
    print()
    print("Next steps:")
    print("  1. Review .env file and modify if needed")
    print("  2. Run: pip install -r requirements.txt")
    print("  3. Run: python run_benchmark.py --fast")
    print()


if __name__ == "__main__":
    main()
