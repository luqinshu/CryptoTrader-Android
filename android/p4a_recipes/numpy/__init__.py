from pythonforandroid.recipe import CompiledComponentsPythonRecipe
from pythonforandroid.logger import shprint, info
from pythonforandroid.util import current_directory
from multiprocessing import cpu_count
from os.path import join
import glob
import sh
import shutil


class NumpyRecipe(CompiledComponentsPythonRecipe):

    version = '1.26.4'
    url = 'https://files.pythonhosted.org/packages/source/n/numpy/numpy-{version}.tar.gz'
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
        """Fix distutils for Python 3.12+ by adding compatibility shim.
        
        Strategy: insert 'import setuptools._distutils as distutils' as the
        very first line of every .py file that references distutils. This makes
        all existing 'distutils.XXX' and 'from distutils.XXX import YYY' work.
        Only special case: msvccompiler (not present in setuptools._distutils).
        """
        import os
        build_dir = self.get_build_dir(arch.arch)

        for root, dirs, files in os.walk(build_dir):
            if '/__pycache__/' in root or '/.git/' in root:
                continue
            for fname in files:
                if not fname.endswith('.py'):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath) as f:
                        content = f.read()
                except Exception:
                    continue

                if 'distutils' not in content:
                    continue

                modified = False

                # Fix 1: msvccompiler (NOT in setuptools._distutils)
                if 'from distutils.msvccompiler import' in content:
                    content = content.replace(
                        'from distutils.msvccompiler import get_build_version as get_build_msvc_version',
                        'try:\n    from distutils.msvccompiler import get_build_version as get_build_msvc_version\nexcept ImportError:\n    get_build_msvc_version = lambda: 14.0'
                    )
                    modified = True

                # Fix 2: Insert compat shim at top of file (after shebang/encoding)
                if 'import setuptools._distutils as distutils' not in content:
                    lines = content.split('\n')
                    # Find the right insertion point: after docstring if any
                    insert_idx = 0
                    # Skip shebang and encoding lines
                    while insert_idx < len(lines) and (
                        lines[insert_idx].startswith('#!') or
                        lines[insert_idx].startswith('# -*-') or
                        lines[insert_idx].startswith('# vim:') or
                        lines[insert_idx].strip() == ''
                    ):
                        insert_idx += 1
                    
                    # Insert the compatibility shim
                    shim_line = 'import setuptools._distutils as distutils'
                    lines.insert(insert_idx, shim_line)
                    content = '\n'.join(lines)
                    modified = True

                if modified:
                    with open(fpath, 'w') as f:
                        f.write(content)
                    info('Fixed distutils in {}'.format(os.path.relpath(fpath, build_dir)))

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
            # Ensure pip is available first
            shprint(hostpython, '-m', 'ensurepip', '--upgrade', '--default-pip', _env=env)
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
