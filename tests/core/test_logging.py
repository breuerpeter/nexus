"""The framework logger's console line, read from the stderr of a script that logs one record."""

import re
import subprocess
import sys


def test_the_console_line_keeps_its_sim_module_prefix(tmp_path):
    """The console line keeps its `[sim/<module>]` prefix.

    Given a script named `console_script.py` with no recording, when it logs a record through
    `nexus_sim.logger`, then its line on stderr carries `[sim/console_script]` before the message.
    """
    script = tmp_path / "console_script.py"
    script.write_text('import nexus_sim as nx\n\nnx.logger.info("hello from the console")\n')
    err = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=True).stderr
    lines = [line for line in err.splitlines() if line.endswith("hello from the console")]

    assert [bool(re.search(r"\[sim/console_script\] INFO hello from the console$", line)) for line in lines] == [True]
