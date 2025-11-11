# Finish PyTorch Upgrade

You currently have an older PyTorch version installed. Complete the upgrade to PyTorch 2.6+ stable:

## Easiest Way (Windows)

**Just run the script:**
```bash
fix_pytorch.bat
```

It will detect and remove your old PyTorch, then install PyTorch 2.9.1 stable with CUDA 12.6.

## Manual Commands (if needed)

**Make sure your virtual environment is activated:**
```bash
venv\Scripts\activate
```

**Run these commands:**
```bash
# Step 1: Uninstall old PyTorch
pip uninstall -y torch torchvision torchaudio

# Step 2: Install PyTorch 2.9.1 stable with CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Step 3: Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

You should see:
```
PyTorch: 2.9.1+cu126
CUDA: True
```

## Test All 16 Models

```bash
python run_benchmark.py --fast
```

All 16 models should now work, including:
- ✅ prajjwal1/bert-tiny
- ✅ prajjwal1/bert-mini
- ✅ prajjwal1/bert-small
- ✅ google/bert_uncased_L-2_H-128_A-2
- ✅ google/bert_uncased_L-4_H-256_A-4
- ✅ google/bert_uncased_L-6_H-512_A-8
- ✅ microsoft/deberta-v3-small
- ✅ microsoft/deberta-v3-base
- ✅ Plus the 8 models that already worked

## About the Compilation Pause

You'll notice a pause after each model loads (5-30 seconds). **This is normal!**

PyTorch 2.5+ and later compile models for optimization on first use. This happens during the warmup phase, so your actual benchmark measurements are unaffected.

Example:
```
Loading bert-base-uncased...
✓ Loaded successfully
[pause ~15 seconds - compilation]
Warmup (10 iterations)...
Testing (100 iterations)...
```

The pause happens once per model and doesn't affect results.

## Done!

After upgrading, you'll have:
- ✅ All 16 models working
- ✅ CVE-2025-32434 security fix
- ✅ NumPy 1.x and 2.x support
- ✅ Optimal inference performance

Delete this file after completing the upgrade.
