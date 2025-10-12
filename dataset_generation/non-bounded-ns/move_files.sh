#!/bin/bash
# Script to organize generated .mat files

echo "Moving generated files to appropriate directories..."

# Move training files (50 files)
echo "Moving training files..."
mv ns-nonbounded_*.mat ../../data/DiffPDE/training/ns-nonbounded_new/ 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ Training files moved to data/DiffPDE/training/ns-nonbounded_new/"
else
    echo "⚠ No training files found (ns-nonbounded_*.mat)"
fi

# Move test file (1 file)
echo "Moving test file..."
mv ns-nonbounded.mat ../../data/DiffPDE/testing/ 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ Test file moved to data/DiffPDE/testing/"
else
    echo "⚠ No test file found (ns-nonbounded.mat)"
fi

echo ""
echo "File organization complete!"
echo "Check the following directories:"
echo "  - data/DiffPDE/training/ns-nonbounded_new/"
echo "  - data/DiffPDE/testing/"

