from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import zipfile

from qec_detector_rtl.scripts.run_sim import SOURCE_FILES, SOURCE_ROOT, digest, validate_vectors


def prepare_bundle(vectors: Path, out_root: Path) -> Path:
    vectors = vectors.resolve()
    manifest = validate_vectors(vectors)
    out_root = out_root.resolve()
    archive_path = Path(str(out_root) + ".zip")
    if archive_path.exists():
        raise FileExistsError(archive_path)
    out_root.mkdir(parents=True, exist_ok=False)
    for name in (*SOURCE_FILES, "scripts/run_vcs.sh"):
        target = out_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE_ROOT / name, target)
    shutil.copyfile(SOURCE_ROOT / "SERVER_RUN.md", out_root / "README.md")
    target_vectors = out_root / "vectors"
    target_vectors.mkdir()
    for name in ["manifest.json", *manifest["files_sha256"]]:
        shutil.copyfile(vectors / name, target_vectors / name)
    validate_vectors(target_vectors)
    files = {
        path.relative_to(out_root).as_posix(): digest(path)
        for path in sorted(out_root.rglob("*")) if path.is_file()
    }
    (out_root / "bundle.json").write_text(json.dumps({
        "status": "prepared_not_run_on_server", "stage": "P2",
        "input_manifest_sha256": digest(vectors / "manifest.json"),
        "files_sha256": files,
    }, indent=2) + "\n", encoding="utf-8")
    files["bundle.json"] = digest(out_root / "bundle.json")
    with (out_root / "CHECKSUMS.sha256").open("w", encoding="ascii", newline="\n") as output:
        for name, checksum in sorted(files.items()):
            output.write(f"{checksum}  {name}\n")
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out_root.rglob("*")):
            if path.is_file():
                archive.write(path, f"{out_root.name}/{path.relative_to(out_root).as_posix()}")
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the P2 source and frozen vectors for the course server")
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_bundle(args.vectors, args.out_root))


if __name__ == "__main__":
    main()
