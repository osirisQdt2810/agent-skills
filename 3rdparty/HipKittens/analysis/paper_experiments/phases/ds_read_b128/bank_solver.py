#!/usr/bin/env python3
"""
Bank Solver for ds_read_b128: Determine the number of LDS banks by testing when addresses wrap around.

Strategy:
- Pick 2 threads from the same phase (so they can conflict)
- Thread 0 reads from banks 0, 1, 2, 3 (offset 0, ds_read_b128 reads 16 bytes = 4 banks)
- Thread 1 reads from progressively higher banks until we detect a conflict
- The first conflict indicates bank wraparound, revealing the number of banks
"""

import torch
import subprocess
import pandas as pd
import os

def run_profiling(should_I_write, offset):
    """Run the kernel with profiling and return conflict information."""

    # Save current test configuration
    test_code = f"""import torch
import tk_kernel

should_I_write = torch.tensor({should_I_write.cpu().tolist()}, dtype=torch.uint32, device='cuda')
offset = torch.tensor({offset.cpu().tolist()}, dtype=torch.uint32, device='cuda')

tk_kernel.dispatch_micro(should_I_write, offset)
"""

    with open('_temp_bank_test.py', 'w') as f:
        f.write(test_code)

    # Run with profiling
    cmd = [
        'rocprofv3',
        '--pmc', 'SQ_INSTS_LDS', 'SQ_LDS_BANK_CONFLICT',
        '--output-format', 'csv',
        '--output-file', 'bank_test',
        '-d', 'out',
        '--',
        'python3', '_temp_bank_test.py'
    ]

    try:
        # Suppress output to keep things clean
        result = subprocess.run(cmd, timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Check if command succeeded
        if result.returncode != 0:
            print(f"\nrocprofv3 failed with return code {result.returncode}")
            return None

    except subprocess.TimeoutExpired:
        print(f"Timeout!")
        return None

    # Read results
    try:
        df = pd.read_csv("out/bank_test_counter_collection.csv")
        # Keep only LDS-related counters
        df = df[df["Counter_Name"].isin(["SQ_INSTS_LDS", "SQ_LDS_BANK_CONFLICT"])]

        # Pivot so each dispatch becomes one row with both counters
        pivot = df.pivot_table(index=["Dispatch_Id", "Kernel_Name"],
                            columns="Counter_Name",
                            values="Counter_Value",
                            aggfunc="first").reset_index()

        # Get the counts from the most recent dispatch (last row)
        last_dispatch = pivot.iloc[-1]
        conflicts = last_dispatch["SQ_LDS_BANK_CONFLICT"]
        lds_insts = last_dispatch["SQ_INSTS_LDS"]
        conflict_ratio = conflicts / lds_insts if lds_insts > 0 else 0

        return lds_insts, conflicts, conflict_ratio
    except Exception as e:
        print(f"\nError reading profiling data: {e}")
        return None
    finally:
        # Cleanup temp file
        if os.path.exists('_temp_bank_test.py'):
            os.remove('_temp_bank_test.py')


def test_bank_conflict(thread_0, thread_1, offset_0, offset_1):
    """Test if two threads have a bank conflict at the given offsets."""

    # Create should_I_write: only thread_0 and thread_1 are enabled
    should_I_write = torch.zeros(64, dtype=torch.uint32, device='cuda')
    should_I_write[thread_0] = 1
    should_I_write[thread_1] = 1

    # Set offsets for each thread
    offset = torch.zeros(64, dtype=torch.uint32, device='cuda')
    offset[thread_0] = offset_0
    offset[thread_1] = offset_1

    result = run_profiling(should_I_write, offset)

    if result is None:
        return None

    lds_insts, conflicts, conflict_ratio = result

    # If there's a conflict, threads are accessing the same bank
    has_conflict = conflicts > 0

    return has_conflict, conflicts, lds_insts


def solve_num_banks():
    """Solve for the number of LDS banks by testing wraparound behavior."""

    print("=" * 70)
    print("Bank Solver (ds_read_b128): Determining the number of LDS banks")
    print("=" * 70)
    print()

    # Use threads 0 and 1 from Phase 0 (they can conflict with each other)
    thread_0 = 0
    thread_1 = 1
    
    print(f"Using threads {thread_0} and {thread_1} (both in Phase 0)")
    print()

    # Thread 0 always reads from offset 0 (banks 0, 1, 2, 3)
    # ds_read_b128 reads 16 bytes (4 floats), so it accesses 4 consecutive banks
    offset_0 = 0
    banks_0 = [0, 1, 2, 3]  # Bank = (byte_offset / 4) % NUM_BANKS
    
    print(f"Thread {thread_0}: offset={offset_0} bytes → banks {banks_0}")
    print()

    # Test thread 1 at progressively higher bank numbers
    # We start at bank 4 to avoid the initial overlap with banks 0, 1, 2, 3
    # ds_read_b128 reads 4 banks, so thread 1 at bank N covers banks N, N+1, N+2, N+3
    
    print("Testing thread 1 at different bank offsets...")
    print()
    print(f"{'Test':<6} {'Bank':<8} {'Offset':<10} {'Conflict':<10} {'Details'}")
    print("-" * 70)

    # We'll test up to bank 128 to be safe (most GPUs have 32 or 64 banks)
    max_bank_to_test = 128
    test_banks = list(range(4, max_bank_to_test, 1))  # Step by 1 to test every bank
    
    first_conflict_bank = None
    conflict_banks = []
    
    for test_num, bank_1 in enumerate(test_banks, 1):
        offset_1 = bank_1 * 4  # Each bank is 4 bytes
        banks_1 = [bank_1, bank_1 + 1, bank_1 + 2, bank_1 + 3]
        
        result = test_bank_conflict(thread_0, thread_1, offset_0, offset_1)
        
        if result is None:
            print(f"{test_num:<6} {bank_1:<8} {offset_1:<10} ERROR")
            continue
        
        has_conflict, conflicts, lds_insts = result
        
        conflict_str = "YES" if has_conflict else "no"
        detail_str = f"banks {banks_1}, conflicts={conflicts}/{lds_insts}"
        
        print(f"{test_num:<6} {bank_1:<8} {offset_1:<10} {conflict_str:<10} {detail_str}")
        
        if has_conflict:
            conflict_banks.append(bank_1)
            if first_conflict_bank is None:
                first_conflict_bank = bank_1
                print()
                print("!" * 70)
                print(f"FIRST CONFLICT DETECTED at bank {bank_1}!")
                print("!" * 70)
                print()
                
                # Determine which bank wrapped around
                # Thread 1 reads banks [bank_1, bank_1+1, bank_1+2, bank_1+3]
                # Thread 0 reads banks [0, 1, 2, 3]
                # Conflict means one of thread 1's banks wrapped to overlap with thread 0's
                
                num_banks = bank_1  # The first bank that wraps to 0
                
                # Verify: bank_1 % num_banks should equal 0, 1, 2, or 3
                wrapped_banks = [b % num_banks for b in banks_1]
                overlap = [b for b in wrapped_banks if b in banks_0]
                
                print(f"Thread 1 banks {banks_1} wrap to {wrapped_banks} (mod {num_banks})")
                print(f"Overlaps with thread 0 banks {banks_0} at: {overlap}")
                print()
                print(f"Continuing to test remaining banks to verify pattern...")
                print()
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if first_conflict_bank is None:
        print(f"WARNING: No conflict detected up to bank {max_bank_to_test}")
        print("Number of banks may be larger than tested range")
        print("=" * 70)
        return None
    
    num_banks = first_conflict_bank
    print(f"Number of LDS banks: {num_banks}")
    print()
    print(f"Conflicts detected at banks: {conflict_banks}")
    print()
    
    # Verify the pattern - conflicts should repeat every num_banks
    print("Verification:")
    expected_conflicts = []
    for bank in conflict_banks:
        # Check which of thread 1's banks (bank, bank+1, bank+2, bank+3) overlap with thread 0's (0, 1, 2, 3)
        wrapped = [(bank + i) % num_banks for i in range(4)]
        overlaps = [b for b in wrapped if b in [0, 1, 2, 3]]
        if overlaps:
            expected_conflicts.append(bank)
            print(f"  Bank {bank}: reads banks {[bank, bank+1, bank+2, bank+3]} → wraps to {wrapped} → overlaps at {overlaps} ✓")
    
    if conflict_banks == expected_conflicts:
        print()
        print("✓ All conflicts match expected pattern!")
    else:
        print()
        print(f"⚠ Unexpected conflicts found")
        print(f"  Expected: {expected_conflicts}")
        print(f"  Got: {conflict_banks}")
    
    print("=" * 70)
    
    # Save results
    with open('bank_results.txt', 'w') as f:
        f.write("LDS Bank Detection Results (ds_read_b128)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Number of LDS banks: {num_banks}\n\n")
        f.write(f"Methodology:\n")
        f.write(f"  - Thread {thread_0} reads from banks {banks_0} (offset {offset_0})\n")
        f.write(f"  - Thread {thread_1} tested progressively higher banks\n")
        f.write(f"  - First conflict at bank {first_conflict_bank}\n")
        f.write(f"  - All conflicts: {conflict_banks}\n")
        f.write(f"  - Confirms {num_banks} banks in LDS\n\n")
        f.write(f"Conflict Pattern:\n")
        for bank in conflict_banks[:10]:  # Show first 10
            wrapped = [(bank + i) % num_banks for i in range(4)]
            overlaps = [b for b in wrapped if b in [0, 1, 2, 3]]
            f.write(f"  Bank {bank}: {[bank, bank+1, bank+2, bank+3]} → {wrapped} → overlap {overlaps}\n")
        if len(conflict_banks) > 10:
            f.write(f"  ... and {len(conflict_banks) - 10} more\n")
    
    print()
    print("Results saved to bank_results.txt")
    return num_banks


if __name__ == '__main__':
    num_banks = solve_num_banks()
    
    if num_banks:
        print()
        print(f"✓ Successfully determined: {num_banks} LDS banks")
    else:
        print()
        print("✗ Could not determine number of banks")

