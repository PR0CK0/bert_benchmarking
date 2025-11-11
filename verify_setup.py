#!/usr/bin/env python
"""
Verify installation and diagnose common issues.
Run this to check if all dependencies are properly installed.
"""

import sys

def check_module(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name

    try:
        __import__(module_name)
        print(f"✓ {package_name} - OK")
        return True
    except ImportError as e:
        print(f"✗ {package_name} - MISSING")
        print(f"  Error: {e}")
        return False

def check_torch():
    """Check PyTorch installation and CUDA support."""
    try:
        import torch
        print(f"✓ PyTorch - OK")
        print(f"  Version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError as e:
        print(f"✗ PyTorch - MISSING")
        print(f"  Error: {e}")
        return False

def check_transformers():
    """Check transformers and test DeBERTa tokenizer (requires protobuf)."""
    try:
        import transformers
        print(f"✓ Transformers - OK")
        print(f"  Version: {transformers.__version__}")

        # Try to load a tokenizer (tests protobuf)
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            print(f"  ✓ Tokenizer loads successfully (protobuf working)")
            return True
        except Exception as e:
            print(f"  ✗ Tokenizer failed (protobuf issue)")
            print(f"    Error: {str(e)[:100]}...")
            print(f"    Solution: pip install protobuf>=3.20.0 and restart terminal")
            return False
    except ImportError as e:
        print(f"✗ Transformers - MISSING")
        print(f"  Error: {e}")
        return False

def main():
    print("="*60)
    print("BERT Benchmarking Suite - Setup Verification")
    print("="*60)
    print()

    all_ok = True

    print("Checking core dependencies:")
    print("-"*60)

    # Check PyTorch
    all_ok &= check_torch()
    print()

    # Check transformers and protobuf
    all_ok &= check_transformers()
    print()

    # Check other core packages
    print("Checking other dependencies:")
    print("-"*60)
    checks = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("yaml", "PyYAML"),
        ("psutil", "psutil"),
        ("tqdm", "tqdm"),
        ("dotenv", "python-dotenv"),
        ("protobuf", "protobuf"),
        ("tabulate", "tabulate"),
        ("sentencepiece", "sentencepiece"),
    ]

    for module, name in checks:
        all_ok &= check_module(module, name)

    print()
    print("="*60)

    if all_ok:
        print("✓ ALL CHECKS PASSED")
        print("="*60)
        print()
        print("Your setup is ready! Run:")
        print("  python run_benchmark.py --fast")
    else:
        print("✗ SOME CHECKS FAILED")
        print("="*60)
        print()
        print("To fix missing dependencies:")
        print("  1. pip install --upgrade -r requirements.txt")
        print("  2. Close and reopen your terminal")
        print("  3. Reactivate virtual environment")
        print("  4. Run this script again")
        print()
        print("For PyTorch with CUDA:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

if __name__ == "__main__":
    main()
