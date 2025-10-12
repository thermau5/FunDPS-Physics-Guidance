#!/usr/bin/env python3
"""
Calculate statistics for the newly generated NS-nonbounded dataset.
Run this AFTER generating the .mat files and moving them to the data directory.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../utils'))

from dataset_prop import calculate_dataset_statistics
import json

if __name__ == "__main__":
    print("="*60)
    print("Calculating statistics for ns-nonbounded_new dataset...")
    print("="*60)
    
    try:
        stats = calculate_dataset_statistics("ns-nonbounded_new")
        
        # Convert numpy arrays to lists for JSON serialization
        stats_serializable = {}
        for key, value in stats.items():
            if hasattr(value, 'tolist'):
                stats_serializable[key] = value.tolist()
            else:
                stats_serializable[key] = value
        
        print("\n" + "="*60)
        print("STATISTICS CALCULATED!")
        print("="*60)
        print("\nPlease update utils/dataset_prop.py with these values:")
        print("\nReplace the placeholder line:")
        print('  "ns-nonbounded_new": {"mean": [0, 0], "std": [1.0, 1.0], ...},')
        print("\nWith:")
        print(f'  "ns-nonbounded_new": {json.dumps(stats_serializable, indent=4)},')
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("  1. Generated the .mat files (run ns_2d.py)")
        print("  2. Moved files to data/DiffPDE/training/ns-nonbounded_new/")
        print("     (run ./move_files.sh)")

