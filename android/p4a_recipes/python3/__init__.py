from pythonforandroid.recipes.python3 import Python3Recipe as BasePython3Recipe


class Python3Recipe(BasePython3Recipe):
    version = "3.11.5"
    url = "https://www.python.org/ftp/python/{version}/Python-{version}.tgz"
    patches = [
        'patches/pyconfig_detection.patch',
        'patches/reproducible-buildinfo.diff',
    ]


recipe = Python3Recipe()
