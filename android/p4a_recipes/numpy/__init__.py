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
        """Fix ALL numpy distutils imports for Python 3.12+ (distutils removed)."""
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
                new_content = content

                # Special: msvccompiler
                if 'from distutils.msvccompiler import' in new_content:
                    new_content = new_content.replace(
                        'from distutils.msvccompiler import get_build_version as get_build_msvc_version',
                        'try:\n    from distutils.msvccompiler import get_build_version as get_build_msvc_version\nexcept ImportError:\n    get_build_msvc_version = lambda: 14.0'
                    )
                    modified = True

                # Generic: replace "from distutils.X" with "from setuptools._distutils.X"
                if 'from distutils.' in new_content:
                    lines = new_content.split('\n')
                    new_lines = []
                    for line in lines:
                        stripped = line.lstrip()
                        if stripped.startswith('from distutils.') and 'msvccompiler' not in stripped:
                            line = line.replace('from distutils.', 'from setuptools._distutils.')
                        elif stripped.startswith('import distutils.') and 'msvccompiler' not in stripped:
                            line = line.replace('import distutils.', 'import setuptools._distutils.')
                        new_lines.append(line)
                    new_content = '\n'.join(new_lines)
                    if new_content != content:
                        modified = True

                if modified:
                    with open(fpath, 'w') as f:
                        f.write(new_content)
                    info('Fixed distutils imports in {}'.format(os.path.relpath(fpath, build_dir)))

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
