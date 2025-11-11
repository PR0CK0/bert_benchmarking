# Quick Start Guide

Complete setup in 6 steps.

## 1. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

## 2. Install PyTorch 2.6+ Stable (GPU Support)

**Windows - Just run:**
```bash
fix_pytorch.bat
```
Done! The script handles everything automatically (installs PyTorch 2.9.1+cu126).

**Linux/Mac:**
```bash
# For CUDA 12.6 (RTX 40xx, newest)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# For CUDA 12.4 (RTX 30xx/40xx)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CUDA 11.8 (older GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Important**: PyTorch 2.6+ stable is required to access all 16 models (fixes CVE-2025-32434).

## 3. Install Other Dependencies

```bash
pip install -r requirements.txt
```

**Verify installation (recommended):**
```bash
python verify_setup.py
```

This checks all dependencies and diagnoses issues. If it passes, you're ready!

## 4. Configure Cache Location

Run the interactive setup:
```bash
python setup_cache.py
```

Or manually create `.env` file:
```bash
# Copy template
cp .env.example .env

# Edit .env and set:
BERT_CACHE_DIR=D:\bert_cache  # Windows
# or
BERT_CACHE_DIR=/mnt/data/bert_cache  # Linux
```

**Important**: Choose a location with 20-50GB free space, NOT in OneDrive/Dropbox.

## 5. Run Fast Benchmark

```bash
python run_benchmark.py --fast
```

First run will download models (~5-15 minutes). Subsequent runs are fast.

## 6. Review Results

**All results are automatically generated!** No extra steps needed.

Results saved to `results/<timestamp>_<type>/` directory:
- `benchmark_results.csv` - All raw metrics
- `benchmark_results.json` - Structured data
- `benchmark_report.md` - Summary report
- `plots_*.png` - Comparison visualizations (4 plots)
- `top_performers_*.png` - Top performers visualizations (4 plots)

**Folder naming**: `YYYYMMDD_HHMMSS_type`
- `_fast` = Fast benchmark (`--fast`)
- `_full` = Full benchmark (default)

**Examples**:
- `results/20251111_101910_fast/` - Fast benchmark run
- `results/20251111_143022_full/` - Full benchmark run

Each benchmark run gets its own timestamped folder with type indicator!

## Quick Commands

```bash
# Test specific models
python run_benchmark.py --models distilbert-base-uncased

# CPU only
python run_benchmark.py --devices cpu

# Full benchmark (all models, all configs)
python run_benchmark.py

# View all available options
python run_benchmark.py --help
```

**For detailed command-line options**, see the [Command-Line Options](README.md#command-line-options) section in the README.

## Troubleshooting

**Long pause after model loads (PyTorch 2.5+)**
- **This is normal!** PyTorch compiles models on first use (5-30 seconds)
- The pause happens during warmup, NOT during actual measurements
- Your benchmark results are accurate - compilation doesn't affect timing
- No action needed

**"ModuleNotFoundError" or "protobuf library not found"**
- Activate virtual environment first
- Run `pip install --upgrade -r requirements.txt`
- **Important**: Close and reopen your terminal after installing packages
- Then reactivate venv and try again

**"No space left on device"**
- Check your cache directory location (in `.env`)
- Ensure 20-50GB free space
- Run `python setup_cache.py` to reconfigure

**Models downloading slowly**
- First run downloads ~5-10GB
- Be patient, subsequent runs reuse cached models
- Check your internet connection

**CUDA out of memory**
- Use CPU: `python run_benchmark.py --devices cpu`
- Or test smaller models first

## What Gets Measured?

- **Latency**: mean, median, p95, p99 (milliseconds)
- **Throughput**: samples per second
- **Memory**: peak usage (MB)
- **Model size**: parameters, disk size

## Recommended First Test

```bash
# Test 3 fast models on CPU only
python run_benchmark.py --fast --devices cpu --models \
  prajjwal1/bert-tiny \
  distilbert-base-uncased \
  sentence-transformers/all-MiniLM-L6-v2
```

This completes in ~5-10 minutes and gives you a good baseline.

## Next Steps

1. Review results in `results/<timestamp>_<type>/` folder
2. Check the report and plots to identify top 2-3 models
3. Test on GPU if available: `--devices cuda`
4. Run full benchmark on promising models (without `--fast`)
5. Create fine-tuning dataset for best models
6. Extend framework to include training benchmarks

## Need Help?

See `README.md` for detailed documentation.
