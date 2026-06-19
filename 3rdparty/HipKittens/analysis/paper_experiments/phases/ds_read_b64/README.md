# ds_read_b64 - LDS Phase and Bank Analysis

This folder contains tools to analyze AMD GPU Local Data Share (LDS) memory behavior using the `ds_read_b64` instruction.

## What is ds_read_b64?

`ds_read_b64` reads **64 bits (8 bytes = 2 floats)** from LDS, accessing **2 consecutive banks** per thread.

## Files

### Core Kernel
- **`kernel.cpp`**: HIP kernel that uses `ds_read_b64` instruction
- **`Makefile`**: Build system for compiling the kernel
- Run `make` to build `tk_kernel.so`

### Solvers

#### 1. Phase Solver (`phase_solver.py`)
Determines which threads are in the same "phase" by testing for bank conflicts.

**Usage:**
```bash
python3 phase_solver.py
```

**Output:** `phase_results.txt` with phase assignments and conflict matrix

**How it works:**
- Tests all 2016 thread pairs (64 choose 2)
- Threads conflict when accessing the same bank → same phase
- Groups threads by conflict patterns

#### 2. Bank Solver (`bank_solver.py`)
Determines the number of LDS banks by detecting wraparound.

**Usage:**
```bash
python3 bank_solver.py
```

**Output:** `bank_results.txt` with bank count and verification

**How it works:**
- Thread 0 reads banks [0, 1] at offset 0
- Thread 1 tests banks 2, 3, 4, ... incrementally
- First conflict indicates wraparound (bank N wraps to 0)
- Continues testing to verify the pattern

### Test Files
- **`quick_test.py`**: Basic sanity tests
- **`test_single.py`**: Single profiling test case
- **`out/analyze_conflicts.py`**: Analyzes profiling results
- **`out/README.md`**: Profiling command examples

## Quick Start

1. **Build the kernel:**
   ```bash
   make
   ```

2. **Test basic functionality:**
   ```bash
   python3 quick_test.py
   ```

3. **Solve for phases** (takes ~30 minutes):
   ```bash
   python3 phase_solver.py
   ```

4. **Solve for bank count** (takes ~5-10 minutes):
   ```bash
   python3 bank_solver.py
   ```

## Instruction Details

- **Instruction:** `ds_read_b64`
- **Size:** 64 bits = 8 bytes = 2 floats
- **Banks accessed:** 2 consecutive banks per thread
- **Bank calculation:** `bank_id = (byte_offset / 4) % NUM_BANKS`

## Results Format

### Phase Results
```
Phase 0: 8 threads - [0, 1, 2, 3, 20, 21, 22, 23]
Phase 1: 8 threads - [4, 5, 6, 7, 16, 17, 18, 19]
...
```

### Bank Results
```
Number of LDS banks: 32 (or 64, depending on GPU)

Conflicts detected at banks: [32, 33, 64, 65, ...]
```

## Dependencies

- PyTorch with CUDA support
- pandas
- rocprofv3 (for profiling)
- HipKittens library

