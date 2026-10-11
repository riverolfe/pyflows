"""Agent calls with a subprocess gate, a disk cache, and a token ceiling.

orch.py is the framework; panel.py renders what it logs. Everything under
examples/ is built on those two and is not part of the framework.
"""

from .orch import Stats, _Spend, ask, gather, read_runs

__all__ = ["Stats", "_Spend", "ask", "gather", "read_runs"]
