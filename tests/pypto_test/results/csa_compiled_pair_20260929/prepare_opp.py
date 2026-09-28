"""Create a writable private OPP root without changing the shared CANN payload."""
import argparse
import os
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    base = args.destination.resolve()
    opp = base / "opp"
    opp.mkdir(parents=True, exist_ok=True)
    for source in (cann / "opp").iterdir():
        if source.name == "static_kernel":
            continue
        destination = opp / source.name
        if not destination.exists():
            destination.symlink_to(source, target_is_directory=source.is_dir())
    platform = base / "aarch64-linux"
    if not platform.exists():
        platform.symlink_to(cann / "aarch64-linux", target_is_directory=True)
    # CANN does not register tiling from the shallow built-in symlink.
    # Materialize this directory chain and its two libraries (about 40 MiB).
    tiling = Path("built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64")
    for relative in [*reversed(tiling.parents[:-1]), tiling]:
        directory = opp / relative
        if directory.is_symlink():
            directory.unlink()
        directory.mkdir(exist_ok=True)
        for source in (cann / "opp" / relative).iterdir():
            target = directory / source.name
            if not target.exists():
                target.symlink_to(source, target_is_directory=source.is_dir())
    for source in (cann / "opp" / tiling).glob("*.so"):
        target = opp / tiling / source.name
        if target.is_symlink():
            target.unlink()
        if not target.exists():
            shutil.copyfile(source, target)
    print(opp)


if __name__ == "__main__":
    main()
