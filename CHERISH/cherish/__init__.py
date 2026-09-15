"""CHERISH model package."""

__all__ = ["MultitaskHER2MILHier"]


def __getattr__(name: str):
    if name == "MultitaskHER2MILHier":
        from .model import MultitaskHER2MILHier

        return MultitaskHER2MILHier
    raise AttributeError(name)
