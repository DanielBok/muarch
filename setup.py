import os

from setuptools import Extension, find_packages, setup

import versioneer

try:
    from Cython.Build import cythonize
    from Cython.Compiler import Options
    from Cython.Distutils.build_ext import new_build_ext as build_ext

    USE_CYTHON = True
except ImportError:
    from distutils.command.build_ext import build_ext

    USE_CYTHON = False


    def cythonize(x, *args, **kwargs):
        return x


    class Options:
        pass

DEV_BUILD = os.environ.get('DEV_BUILD', '0').lower() in ('true', '1')
PACKAGE_NAME = 'muarch'

cmdclass = versioneer.get_cmdclass()
cmdclass.update({
    'build_ext': build_ext
})

with open('README.md') as f:
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
            m['sources'][i] = _file + ext

        extensions.append(Extension(**m, language=language, define_macros=macros))

    # compiler options
    Options.annotate = DEV_BUILD

    # compiler directives
    directives = {'language_level': '3', 'profile': DEV_BUILD, 'linetrace': DEV_BUILD}

    return cythonize(extensions, compiler_directives=directives)


install_requires = [
    'arch >=4.7',
    'copulae >=0.4.0',
    'numpy >=1.15',
    'scipy >=1.1',
    'setuptools >=40.8',
    'pandas >=0.23'
]

setup(
    name=PACKAGE_NAME,
    license='MIT',
    version=versioneer.get_version().split('+')[0],
    description='Multiple Univariate ARCH modeling toolbox built on top of the ARCH package',
    author='Daniel Bok',
    author_email='daniel.bok@outlook.com',
    packages=find_packages(include=['muarch', 'muarch.*']),
    ext_modules=build_ext_modules(),
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
    ],
    install_requires=install_requires,
    extras_require={"plots": ["matplotlib"]},
    python_requires='>=3.7',
    include_package_data=True,
    zip_safe=False
)
