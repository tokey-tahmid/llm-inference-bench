"""Backend-agnostic LLM inference performance characterisation harness.

See ../CLAUDE.md for the non-negotiable measurement rules. The one that shapes
this package: nothing under ``results/raw/`` is ever produced by anything but a
real run, and ``provenance.write_artifact`` is the only function that writes there.
"""

__version__ = "0.1.0"
