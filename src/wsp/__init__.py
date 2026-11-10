"""wsp: a small harness for hand-editing model weights. See README.md."""

import logging
import os

# Keep tool output readable for an agent: no progress bars or library chatter.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

from .workspace import Workspace  # noqa: E402,F401
