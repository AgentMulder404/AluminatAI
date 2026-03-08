"""
AluminatAI CLI entry point.

When installed via `pip install aluminatiai`, the `aluminatiai` command
runs this module.  sys.path is patched first so the bare-import modules
(collector, config, uploader, etc.) resolve against the installed package
directory regardless of the working directory.
"""
import os
import sys


def main() -> None:
    # Insert the package directory at the front of sys.path so that
    # bare imports like `from collector import GPUCollector` resolve to
    # the modules installed alongside this file (site-packages/aluminatiai/).
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    # Delegate to agent.main() — it owns argparse + the run loop.
    from agent import main as _main  # noqa: PLC0415
    sys.exit(_main())


if __name__ == "__main__":
    main()
