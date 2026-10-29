"""entroptics-llm is now entroptics-context; this module re-exports it."""
import warnings

warnings.warn("entroptics_llm is now entroptics_context; import entroptics_context instead.",
              DeprecationWarning, stacklevel=2)

from entroptics_context import *  # noqa: E402,F401,F403
from entroptics_context import __version__  # noqa: E402,F401
