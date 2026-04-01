#!/usr/bin/env python3
"""Print SLURM job / run name from config: fno+dataset+bs+lr+ep_m (for use by run_fno.slurm)."""
import sys
import yaml


def main():
    config_path = sys.argv[1]
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    ds = cfg["dataset_name"]
    bs = cfg["batch_size"]
    lr = cfg["learning_rate"]
    ep = cfg["num_epochs"]
    modes = cfg["fno_modes"]
    m = modes[0] if modes else ""
    lr_s = str(lr).replace(".", "_")
    runname = f"fno-{ds}-bs{bs}-lr{lr_s}-ep{ep}_m{m}"
    print(runname)


if __name__ == "__main__":
    main()
