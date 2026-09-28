"""The stand-in component class, in a module of its own, so a test can tell whether the build imported it."""


class StandIn:
    """A component that records its keyword arguments."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
