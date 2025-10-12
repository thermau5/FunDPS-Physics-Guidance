# NS-Nonbounded New Dataset Generation Guide

This guide walks you through generating a new NS 2D dataset with updated parameters (25th → 50th timesteps).

## 📋 Overview

- **Training**: 50,000 samples (50 files × 1,000 samples each)
- **Testing**: 1,000 samples (1 file)
- **Time steps**: 50 (extracting 25th and 50th for input/output)
- **Resolution**: 128×128
- **Output**: HuggingFace Arrow datasets
- **File size**: ~262 MB per file (optimized to avoid MATLAB format limits)

---

## 🚀 Step-by-Step Instructions

### **Step 1: Generate Training Data (50,000 samples)**

```bash
cd dataset_generation/non-bounded-ns
python ns_2d.py
```

**Expected output**: 50 files named `ns-nonbounded_1.mat` to `ns-nonbounded_50.mat`

**Time estimate**: ~10-20 hours depending on GPU

---

### **Step 2: Generate Testing Data (1,000 samples)**

```bash
cd dataset_generation/non-bounded-ns
python ns_2d_test.py
```

**Expected output**: 1 file named `ns-nonbounded.mat`

**Time estimate**: ~15-30 minutes

---

### **Step 3: Organize Generated Files**

```bash
cd dataset_generation/non-bounded-ns
./move_files.sh
```

This will move:
- Training files → `data/DiffPDE/training/ns-nonbounded_new/`
- Test file → `data/DiffPDE/testing/`

---

### **Step 4: Calculate Statistics**

```bash
cd dataset_generation/non-bounded-ns
python calculate_stats.py
```

**Output**: Mean, std, min, max for both channels

**Action required**: Copy the output and update the `STATS` dictionary in `utils/dataset_prop.py` at line 15.

Replace:
```python
"ns-nonbounded_new": {"mean": [0, 0], "std": [1.0, 1.0], ...},  # PLACEHOLDER
```

With the calculated values.

---

### **Step 5: Process into HuggingFace Datasets**

```bash
# Process training data (50,000 samples)
python utils/dataset_process.py ns-nonbounded_new

# Process testing data (1,000 samples)
python utils/dataset_process.py ns-nonbounded_new --test
```

**Output directories**:
- `data/DiffPDE/ns-nonbounded_new_hf/` (training)
- `data/DiffPDE/ns-nonbounded_test_new_hf/` (testing)

---

## 📊 Data Format

### Raw .mat files contain:
- `u`: Selected time evolution (shape: [N, 128, 128, 2])
  - `u[..., 0]`: 25th timestep
  - `u[..., 1]`: 50th timestep
- `t`: Time values (shape: [2])

### HuggingFace datasets contain:
- `id`: Sample index
- `data`: Shape [2, 128, 128]
  - `data[0]`: 25th timestep (input)
  - `data[1]`: 50th timestep (output)

---

## ✅ Verification

After processing, check the datasets:

```python
from datasets import load_from_disk

# Load training data
train_ds = load_from_disk("data/DiffPDE/ns-nonbounded_new_hf")
print(f"Training samples: {len(train_ds)}")  # Should be 50,000
print(f"Sample shape: {train_ds[0]['data'].shape}")  # Should be (2, 128, 128)

# Load test data
test_ds = load_from_disk("data/DiffPDE/ns-nonbounded_test_new_hf")
print(f"Test samples: {len(test_ds)}")  # Should be 1,000
```

---

## 🔧 Troubleshooting

### Out of Memory during generation:
- Reduce `bsize` from 200 to 100 or 50 in `ns_2d.py`

### Files not found:
- Check that `move_files.sh` ran successfully
- Verify files are in the correct directories

### Statistics calculation fails:
- Ensure all 50 training files are present
- Check file paths match the configuration

---

## 📝 Key Differences from Old Dataset

| Parameter | Old | New |
|-----------|-----|-----|
| **Input** | Initial condition (`a`) | 25th timestep |
| **Output** | Last timestep | 50th timestep |
| **Total timesteps** | 10-50 | 50 |
| **Samples** | Variable | 50,000 + 1,000 |
| **File format** | Full trajectory | Only 2 timesteps (efficient!) |

---

## 🎯 Next Steps

After verifying the new datasets work correctly:

1. Update your training scripts to use `ns-nonbounded_new_hf`
2. Compare results with old dataset
3. If satisfied, optionally replace old dataset or keep both

---

## 📞 Need Help?

Check the following files for configuration:
- `utils/dataset_prop.py` - Dataset loaders and paths
- `utils/dataset_process.py` - Processing pipeline
- `ns_2d.py` / `ns_2d_test.py` - Generation parameters

