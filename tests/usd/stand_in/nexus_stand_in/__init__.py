"""The stand-in project's schema plugin, `StandInAPI`, registered on import as Newton registers its own."""

import pathlib

from pxr import Plug

Plug.Registry().RegisterPlugins([pathlib.Path(__file__).parent.absolute().as_posix()])
