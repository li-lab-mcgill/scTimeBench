# __init__.py
import pkgutil
import importlib

# Automatically import all modules in the current package to trigger registration
for loader, module_name, is_pkg in pkgutil.walk_packages(
    __path__, prefix=f"{__name__}."
):
    importlib.import_module(module_name)
