# Backwards-compat shim. #20646 moved network / socket helpers out of
# `sglang.srt.utils.common` into a new `sglang.srt.utils.network` submodule
# but did not re-export them from this package's top level, so external
# consumers still doing `from sglang.srt.utils import is_port_available`
# (like sglang-router <= 0.3.2's launch_server.py) hit ImportError.
from sglang.srt.utils.common import *  # noqa: F401,F403
from sglang.srt.utils.network import is_port_available  # noqa: F401
