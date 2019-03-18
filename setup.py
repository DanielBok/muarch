import os
import sys

from setuptools import Extension, find_packages, setup

import versioneer

try:
    from Cython.Build import cythonize, build_ext as _build_cython_ext
    from Cython.Compiler import Options

    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False


    def cythonize(x, *args, **kwargs):
        return x


    def _build_cython_ext(*args, **kwargs):
        return None


    class Options:
        pass

DEV_BUILD = os.environ.get('DEV_BUILD', '0').lower() in ('true', '1')
PACKAGE_NAME = 'muarch'

cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = _build_cython_ext

with open('readme.md') as f:
    long_description = f.read()


def build_ext_modules():
    macros = [('NPY_NO_DEPRECATED_API', '1'),
              ('NPY_1_7_API_VERSION', '1')]

    modules = [
        {
            'name': 'muarch.volatility._vol_simulations',
            'sources': ['muarch/volatility/_vol_simulations.pyx'],
        },
        {
            'name': 'muarch.mean._mean_simulations',
            'sources': ['muarch/mean/_mean_simulations.pyx'],
        }
    ]

    extensions = []
    for m in modules:
        # if not built with Cython, use the c or cpp files
        language = m.get('language', 'c')
        ext = '.pyx' if USE_CYTHON else f'.{language}'

        for i, source in enumerate(m['sources']):
            _file, _ext = os.path.splitext(source)
            if ext in ('.pyx', '.py'):
                m['sources'][i] = _file + ext

        extensions.append(Extension(**m, language=language, define_macros=macros))

    # compiler options
    Options.annotate = DEV_BUILD

    # compiler directives
    directives = {'language_level': '3', 'profile': DEV_BUILD, 'linetrace': DEV_BUILD}

    return cythonize(extensions, compiler_directives=directives)


def run_setup():
    ext_modules = build_ext_modules()

    major, minor, *_ = sys.version_info

    if major != 3:
        raise RuntimeError('Please build on python 3!')

    setup_requires = [
        'cython >=0.29',
        'numpy >=1.15'
    ]

    install_requires = [
        'arch >=4.7',
        'copulae >=0.2.0',
        'numpy >= 1.15',
        'scipy >=1.1',
        'pandas >=0.23'
    ]

    setup(
        name=PACKAGE_NAME,
        license='MIT',
        version=versioneer.get_version(),
        description='Multiple Univariate ARCH modeling toolbox built on top of the ARCH package',
        author='Daniel Bok',
        author_email='daniel.bok@outlook.com',
        packages=find_packages(include=['muarch', 'muarch.*']),
        ext_modules=ext_modules,
        url='https://github.com/DanielBok/muarch',
        cmdclass=cmdclass,
        long_description=long_description,
        long_description_content_type='text/markdown',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Education',
            'Intended Audience :: End Users/Desktop',
            'Intended Audience :: Financial and Insurance Industry',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Programming Language :: Cython',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Topic :: Education',
            'Topic :: Scientific/Engineering',
        ],
        install_requires=install_requires,
        setup_requires=setup_requires,
        python_requires='>=3.6',
        include_package_data=True,
        zip_safe=False
    )


run_setup()
