# measure bank conflicts
rocprofv3 --pmc SQ_INSTS_LDS SQ_LDS_BANK_CONFLICT --output-format csv --output-file lds_conflict -d out -- python3 test_single.py

# view bank conflicts
python3 out/analyze_conflicts.py

# measure L1 and L2 cache hit rate
rocprofv3 --pmc TCP_TOTAL_CACHE_ACCESSES_sum --output-format csv --output-file hit_rate -d out -- python3 test_single.py

# view hit rate
python3 out/analyze_hitrate.py

