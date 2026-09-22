"""Private preparation scratch and recoverable publication of input bundles."""

import os
import shutil
import tempfile
from pathlib import Path


def temporary_directory(prefix: str):
    """Prefer node-local Slurm scratch, otherwise respect tempfile/TMPDIR."""
    base = os.environ.get("SLURM_TMPDIR")
    return tempfile.TemporaryDirectory(
        prefix=prefix, dir=base if base and Path(base).is_dir() else None,
    )


def publish_bundle(stage: Path, destination: Path, json_name: str | None = None) -> None:
    """Publish validated files, JSON last; roll back caught publication failures.

    This is not a cross-process snapshot transaction. One writer owns a target
    bundle; inference should not read that bundle during its publication.
    """
    files = sorted(path for path in stage.rglob("*") if path.is_file())
    files.sort(key=lambda path: path.relative_to(stage).as_posix() == json_name)
    backups = []
    replaced = []
    pending = []
    preserve_backups = False
    try:
        for source in files:
            target = destination / source.relative_to(stage)
            target.parent.mkdir(parents=True, exist_ok=True)
            backup = None
            if target.exists():
                backup_dir = Path(tempfile.mkdtemp(prefix=".boltz-backup-", dir=target.parent))
                backups.append(backup_dir)
                backup = backup_dir / "original"
                try:
                    os.link(target, backup)
                except OSError:
                    shutil.copy2(target, backup)
            with tempfile.NamedTemporaryFile(prefix=".boltz-publish-", dir=target.parent, delete=False) as handle:
                temporary = Path(handle.name)
            pending.append(temporary)
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
            replaced.append((target, backup))
    except BaseException as error:
        rollback_errors = []
        for target, backup in reversed(replaced):
            try:
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    os.replace(backup, target)
            except OSError as rollback_error:
                rollback_errors.append(str(rollback_error))
        if rollback_errors:
            preserve_backups = True
            raise RuntimeError(
                f"Prepared publication and rollback failed: {rollback_errors}. "
                f"Recovery backups retained at {backups}"
            ) from error
        raise
    finally:
        for temporary in pending:
            temporary.unlink(missing_ok=True)
        if not preserve_backups:
            for backup_dir in backups:
                shutil.rmtree(backup_dir)
