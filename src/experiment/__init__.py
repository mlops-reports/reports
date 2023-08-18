"""Report AI module"""
from pathlib import Path

NAME = "experiment"
__build__ = {
    line.strip().split()[-1]: line.strip().split()[0]
    for line in (
        (Path(__file__).parent / ".build").open()
        if (Path(__file__).parent / ".build").exists()
        else []
    )
}.get("HEAD", None)
__all__ = ["api", "utils"]
