# Upgrading to PyTorch 2.6+ (Stable)

This guide helps you upgrade to PyTorch 2.6+ stable release to fix CVE-2025-32434 and enable all 16 models.

## Why Upgrade?

PyTorch 2.6.0 and later (stable releases) fix a critical security vulnerability (CVE-2025-32434) that affects loading models in pickle format. After upgrading, you'll be able to use:

- All prajjwal1/bert-* models (tiny, mini, small)
- All google/bert_uncased_* models
- All microsoft/deberta-v3-* models

**Total: 16 models** instead of just 7!

## Is It Safe?

**Yes!** PyTorch 2.6+ is backward compatible with PyTorch 2.1.x. Your existing code won't break.

**What about dependencies?**
- ✅ **Pandas**: No changes needed - pandas doesn't depend on PyTorch
- ✅ **NumPy**: PyTorch 2.6+ supports both NumPy 1.x and 2.x
- ✅ **Transformers**: Compatible with PyTorch 2.6+
- ✅ **Other packages**: All compatible

## Step-by-Step Upgrade

### Option 1: Windows (Automated)

**With virtual environment activated:**

```batch
fix_pytorch.bat
```

This script will:
1. Uninstall your current PyTorch
2. Upgrade NumPy
3. Install PyTorch 2.9.1 stable with CUDA 12.6

### Option 2: Manual Installation

**Step 1: Uninstall old PyTorch**
```bash
pip uninstall -y torch torchvision torchaudio
```

**Step 2: Upgrade NumPy**
```bash
pip install --upgrade "numpy>=1.24.0"
```

**Step 3: Install PyTorch 2.6+ with CUDA**

PyTorch 2.6+ is now stable and officially released. Choose the CUDA version that matches your system:

```bash
# For CUDA 12.6 (RTX 40xx series, newest) - RECOMMENDED
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# For CUDA 12.4 (RTX 30xx/40xx series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CUDA 11.8 (older GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Note**: These are stable releases, not nightly builds. Latest version is 2.9.1+.

**Step 4: Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

You should see:
```
PyTorch: 2.9.1+cu126  (or similar stable version)
CUDA: True
```

## Expected Behavior After Upgrade

### First-Inference Compilation Pause

**You may notice a pause after "Loading model..." before inference starts.** This is normal!

PyTorch 2.5+ includes optimizations that compile models on first use:
- **What you'll see**: Model loads → pause (5-30 seconds) → inference starts fast
- **Why it happens**: PyTorch is JIT-compiling and optimizing the model for your hardware
- **This only happens once per model** - subsequent runs are fast
- **The warmup phase in benchmarking handles this** - real inference timing is unaffected

**During benchmarks:**
```
Loading bert-base-uncased...
✓ Loaded successfully
[pause here - compilation happening]
Warmup (10 iterations)...  ← This absorbs the compilation time
Testing (100 iterations)...  ← These are the real measurements
```

The compilation pause doesn't affect your benchmark results because it happens during warmup.

## Test the Upgrade

Run a fast benchmark to verify all models work:

```bash
python run_benchmark.py --fast --devices cpu
```

You should now see **all 16 models** benchmarking successfully, including:
- prajjwal1/bert-tiny ✓
- prajjwal1/bert-mini ✓
- prajjwal1/bert-small ✓
- google/bert_uncased_L-2_H-128_A-2 ✓
- google/bert_uncased_L-4_H-256_A-4 ✓
- google/bert_uncased_L-6_H-512_A-8 ✓
- microsoft/deberta-v3-small ✓
- microsoft/deberta-v3-base ✓

## Troubleshooting

### "No module named 'torch'"
Your virtual environment may not be activated:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### "CUDA available: False" after upgrade
You installed the CPU-only version. Make sure you used the `--index-url` flag for CUDA:
```bash
# For CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# For CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### NumPy compatibility errors
If you see NumPy errors:
```bash
pip install --upgrade numpy
```

PyTorch 2.6+ works with both NumPy 1.x and 2.x.

### Still getting CVE-2025-32434 errors
Your PyTorch version is too old (or a dev build from before 2.6.0 stable):
```bash
# Check version
python -c "import torch; print(torch.__version__)"

# If < 2.6.0 (or shows 2.6.0.dev from 2024), reinstall:
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**Important**: Dev/nightly builds from before the stable 2.6.0 release (e.g., `2.6.0.dev20241112`) will NOT work. You need stable 2.6.0 or later.

## Rolling Back (if needed)

If you need to rollback for any reason:

```bash
# Uninstall PyTorch 2.6+
pip uninstall -y torch torchvision torchaudio

# Reinstall PyTorch 2.5.x (latest before 2.6)
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Optionally downgrade NumPy if needed
pip install "numpy<2.0"
```

Note: After rollback, only the 7 safetensors-based models will work.

## Summary

- **Risk**: Very low - PyTorch 2.6+ is backward compatible
- **Benefit**: Access to 16 models instead of 7
- **Time**: 2-5 minutes to upgrade
- **Recommendation**: Upgrade! The security fix and model access are worth it

After upgrading, you'll have access to the full suite of tiny, small, and efficient models for better benchmarking coverage!
