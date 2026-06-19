#!/usr/bin/env python3
"""
Measure compile time for all kernels in HipKittens/kernels.

This script finds all Makefiles, runs make clean and make, and records compile times.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Tuple
import sys

def find_makefiles(root_dir: str) -> List[Tuple[str, str]]:
    """
    Find all Makefiles and return (directory_path, relative_path) tuples.
    Excludes micros and archived kernels.
    """
    makefiles = []
    root_path = Path(root_dir)
    
    for makefile_path in root_path.rglob("Makefile"):
        dir_path = makefile_path.parent
        rel_path = dir_path.relative_to(root_path)
        rel_path_str = str(rel_path)
        
        # Skip micros, archived kernels, torch_scaled, and FP8 8wave
        if ("micros" in rel_path_str.lower() or 
            "archive" in rel_path_str.lower() or
            "torch_scaled" in rel_path_str.lower() or
            "8wave" in rel_path_str.lower()):
            continue
        
        makefiles.append((str(dir_path), rel_path_str))
    
    return sorted(makefiles)

def measure_compile_time(makefile_dir: str, relative_path: str, verbose: bool = False) -> Dict:
    """
    Measure compile time for a single kernel.
    Returns dict with timing info and status.
    """
    result = {
        "path": relative_path,
        "status": "unknown",
        "compile_time": None,
        "error": None
    }
    
    original_dir = os.getcwd()
    
    try:
        os.chdir(makefile_dir)
        
        # Clean first
        if verbose:
            print(f"  Cleaning {relative_path}...")
        clean_start = time.time()
        clean_result = subprocess.run(
            ["make", "clean"],
            capture_output=True,
            text=True,
            timeout=60
        )
        clean_time = time.time() - clean_start
        
        # Measure compile time
        if verbose:
            print(f"  Compiling {relative_path}...")
        compile_start = time.time()
        
        compile_result = subprocess.run(
            ["make", "-j1"],  # Use -j1 to get accurate single-threaded compile time
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        compile_time = time.time() - compile_start
        
        if compile_result.returncode == 0:
            result["status"] = "success"
            result["compile_time"] = compile_time
            result["clean_time"] = clean_time
        else:
            result["status"] = "failed"
            result["compile_time"] = compile_time
            result["error"] = compile_result.stderr[:500]  # First 500 chars of error
            
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Compilation exceeded 1 hour timeout"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    finally:
        os.chdir(original_dir)
    
    return result

def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Measure compile time for all kernels")
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "kernels"),
        help="Root directory to search for Makefiles (default: HipKittens/kernels)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output during compilation"
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Only measure kernels matching this path pattern (e.g., 'attn/gqa')"
    )
    
    args = parser.parse_args()
    
    # Check if THUNDERKITTENS_ROOT is set
    if "THUNDERKITTENS_ROOT" not in os.environ:
        print("WARNING: THUNDERKITTENS_ROOT environment variable is not set.")
        print("Some kernels may fail to compile.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    makefiles = find_makefiles(args.root)
    
    if args.filter:
        makefiles = [(d, p) for d, p in makefiles if args.filter in p]
    
    if not makefiles:
        return
    
    for i, (makefile_dir, relative_path) in enumerate(makefiles, 1):
        print(f"[{i}/{len(makefiles)}] {relative_path}")
        result = measure_compile_time(makefile_dir, relative_path, verbose=args.verbose)
        
        if result["status"] == "success":
            print(f"  Success ({format_time(result['compile_time'])})")
        elif result["status"] == "failed":
            print(f"  Failed ({format_time(result.get('compile_time', 0))})")
            if args.verbose and result.get("error"):
                print(f"    Error: {result['error'][:200]}")
        else:
            print(f"  {result['status']}")

if __name__ == "__main__":
    main()
