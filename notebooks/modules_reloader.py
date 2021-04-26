from importlib import reload


def reloader(*modules):
    """Usage:

    >>> from reload_modules import reloader
    >>> from modules import a, b, c  # modules to be dynamically updated parallel to notebook's usage
    >>> r = reloader(a, b, c)
    ...
    >>> r()  # when something has changed in one of the a, b, c modules
    """

    def reload_():
        for module in modules:
            reload(module)
    return reload_
