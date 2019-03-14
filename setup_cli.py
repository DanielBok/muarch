# run this with
# python setup_cli.py develop
from setuptools import find_packages, setup

import versioneer

AUTHOR = 'Daniel Bok'
EMAIL = 'daniel.bok@outlook.com'

setup(
    name='muarch-cli',
    author=AUTHOR,
    author_email=EMAIL,
    maintainer=AUTHOR,
    maintainer_email=EMAIL,
    version=versioneer.get_version(),
    install_requires=[
        'Click'
    ],
    include_package_data=True,
    packages=find_packages(include=['cli', 'cli.*']),
    entry_points="""
        [console_scripts]
        muarch=cli.main:main
    """
)
