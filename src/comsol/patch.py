import platform
from mph.discovery import architectures


def patch():
    arch = platform.machine()
    system = platform.system()

    # Fix for macarm64 planform
    # See https://github.com/MPh-py/MPh/issues/80
    if system == "Darwin" and arch == "arm64":
        architectures[system].append("macarm64")
