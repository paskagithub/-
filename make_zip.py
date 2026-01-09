import argparse
from pathlib import Path
import zipfile


FILES = [
    "ot_instances.py",
    "metrics.py",
    "lp_gurobi.py",
    "lp_mosek.py",
    "admm_ot.py",
    "sinkhorn.py",
    "dual_grad_eot.py",
    "run_all.py",
    "README.md",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create submission zip.")
    parser.add_argument("--name", default="YOURNAME")
    parser.add_argument("--id", default="YOURID")
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{args.name}-{args.id}.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for filename in FILES:
            path = Path(filename)
            if path.exists():
                zipf.write(path, arcname=path.name)
            else:
                print(f"Warning: missing {filename}")

    print(str(zip_path))


if __name__ == "__main__":
    main()
