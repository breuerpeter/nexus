"""The installed command-line tool: the console script the distribution puts on the path."""

import shutil
import subprocess


def test_the_command_line_tool_runs_under_the_project_name():
    """The renamed package still installs and runs: `nexus --help` works.

    The console script is what someone who installed the package types, so the test runs the command
    the path resolves and reads what it prints, rather than importing the parser.
    """
    exe = shutil.which("nexus")
    assert exe, "no `nexus` command on the path: the package installs under another name"
    out = subprocess.run([exe, "--help"], check=False, capture_output=True, text=True)
    assert (out.returncode, out.stdout.startswith("usage: nexus")) == (0, True)
