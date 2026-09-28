"""Create a writable private OPP root without changing the shared CANN payload."""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]


def main():
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    base = REPO / ".cache/csa/native-template-cann92-opp"
    opp = base / "opp"
    opp.mkdir(parents=True, exist_ok=True)
    for source in (cann / "opp").iterdir():
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
