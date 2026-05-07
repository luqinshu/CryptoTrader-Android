from pythonforandroid.recipes.hostpython3 import HostPython3Recipe as BaseHostPython3Recipe


class HostPython3Recipe(BaseHostPython3Recipe):
    version = "3.11.5"
    url = "https://www.python.org/ftp/python/{version}/Python-{version}.tgz"
    patches = []


recipe = HostPython3Recipe()
