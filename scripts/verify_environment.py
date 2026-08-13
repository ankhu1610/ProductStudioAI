"""Validate the local ProductStudio development environment without loading a model."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from dataclasses import dataclass


REQUIRED_PACKAGES = ("torch", "diffusers", "transformers", "fastapi", "gradio")


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def package_check(package: str) -> Check:
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return Check(package, False, "not installed")
    return Check(package, True, version)


def torch_check() -> list[Check]:
    try:
        import torch
    except ImportError:
        return [Check("PyTorch CUDA", False, "torch is not installed")]

    checks = [Check("PyTorch", True, torch.__version__)]
    cuda_available = torch.cuda.is_available()
    checks.append(Check("CUDA available", cuda_available, str(cuda_available)))
    if cuda_available:
        checks.append(Check("CUDA runtime", True, str(torch.version.cuda)))
        checks.append(Check("GPU", True, torch.cuda.get_device_name(0)))
        checks.append(Check("GPU VRAM", True, f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB"))
    else:
        checks.append(Check("GPU", False, "CUDA GPU not detected; generation will be CPU-only"))
    return checks


def main() -> int:
    checks = [
        Check("Python", sys.version_info >= (3, 10), platform.python_version()),
        Check("Platform", True, platform.platform()),
        *(package_check(package) for package in REQUIRED_PACKAGES if package != "torch"),
        *torch_check(),
    ]
    print("ProductStudio AI environment verification")
    print("-" * 48)
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    passed = all(check.ok for check in checks)
    if not passed:
        print("\nFix failed checks before proceeding to Phase 1. See README.md and data/README_SETUP.md.")
        return 1
    print("\nEnvironment is ready for Phase 1 development.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

