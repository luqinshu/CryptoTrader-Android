from pythonforandroid.recipe import CompiledComponentsPythonRecipe
from pythonforandroid.logger import shprint, info
from pythonforandroid.util import current_directory
from multiprocessing import cpu_count
from os.path import join
import glob
import sh
import shutil


class NumpyRecipe(CompiledComponentsPythonRecipe):

    version = '1.22.3'
    url = 'https://pypi.python.org/packages/source/n/numpy/numpy-{version}.zip'
    site_packages_name = 'numpy'
    depends = ['setuptools']
    install_in_hostpython = True
    call_hostpython_via_targetpython = False

    patches = [
        join("patches", "remove-default-paths.patch"),
        join("patches", "add_libm_explicitly_to_build.patch"),
        join("patches", "ranlib.patch"),
    ]

    def _fix_distutils_import(self, arch):
        """Fix numpy's distutils imports for Python 3.12+."""
        import os
        build_dir = self.get_build_dir(arch.arch)

        # Fix 1: msvccompiler (not in setuptools._distutils)
        target = os.path.join(build_dir, 'numpy', 'distutils', 'mingw32ccompiler.py')
        if os.path.exists(target):
            with open(target) as f:
                content = f.read()
            old = 'from distutils.msvccompiler import get_build_version as get_build_msvc_version'
            if old in content:
                new = 'try:\n    from distutils.msvccompiler import get_build_version as get_build_msvc_version\nexcept ImportError:\n    get_build_msvc_version = lambda: 14.0'
                content = content.replace(old, new)
                with open(target, 'w') as f:
                    f.write(content)
                info('Fixed msvccompiler import')

        # Fix 2: distutils.version (available in setuptools._distutils)
        target = os.path.join(build_dir, 'tools', 'cythonize.py')
        if os.path.exists(target):
            with open(target) as f:
                content = f.read()
            old = 'from distutils.version import LooseVersion'
            if old in content:
                new = 'from setuptools._distutils.version import LooseVersion'
                content = content.replace(old, new)
                with open(target, 'w') as f:
                    f.write(content)
                info('Fixed LooseVersion import in cythonize.py')

    def get_recipe_env(self, arch=None, with_flags_in_cc=True):
        env = super().get_recipe_env(arch, with_flags_in_cc)
        env["_PYTHON_HOST_PLATFORM"] = arch.command_prefix
        env["NPY_DISABLE_SVML"] = "1"
        return env

    def _build_compiled_components(self, arch):
        info('Building compiled components in {}'.format(self.name))
        self._fix_distutils_import(arch)
        env = self.get_recipe_env(arch)
        with current_directory(self.get_build_dir(arch.arch)):
            hostpython = sh.Command(self.hostpython_location)
            # Ensure setuptools is installed (provides _distutils for Python 3.12+)
            shprint(hostpython, '-m', 'pip', 'install', 'setuptools', '-q', _env=env)
            shprint(hostpython, 'setup.py', self.build_cmd, '-v', _env=env, *self.setup_extra_args)
            build_dir = glob.glob('build/lib.*')[0]
            shprint(sh.find, build_dir, '-name', '"*.o"', '-exec', env['STRIP'], '{}', ';', _env=env)

    def _rebuild_compiled_components(self, arch, env):
        info('Rebuilding compiled components in {}'.format(self.name))
        hostpython = sh.Command(self.real_hostpython_location)
        shprint(hostpython, 'setup.py', 'clean', '--all', '--force', _env=env)
        shprint(hostpython, 'setup.py', self.build_cmd, '-v', _env=env, *self.setup_extra_args)

    def build_compiled_components(self, arch):
        self.setup_extra_args = ['-j', str(cpu_count())]
        self._build_compiled_components(arch)
        self.setup_extra_args = []

    def rebuild_compiled_components(self, arch, env):
        self.setup_extra_args = ['-j', str(cpu_count())]
        self._rebuild_compiled_components(arch, env)
        self.setup_extra_args = []

    def get_hostrecipe_env(self, arch=None):
        env = super().get_hostrecipe_env(arch)
        env['RANLIB'] = shutil.which('ranlib')
        return env


recipe = NumpyRecipe()
