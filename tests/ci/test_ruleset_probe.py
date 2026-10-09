"""Throwaway probe for #215: a red gpu-pytest must block the merge. Never merge this file."""

import pytest


@pytest.mark.gpu
def test_red_gpu_leg_blocks_merge():
    assert False, "deliberate failure: proves a red gpu-pytest blocks the merge"
