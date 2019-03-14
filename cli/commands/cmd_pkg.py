import shutil
from pathlib import Path

import click

from cli.utils import *


@click.group()
def cli():
    """Tools to build and manage Copulae packaging"""
    pass


@cli.command()
@click.option('--no-clean', 'no_clean', is_flag=True, default=True, help='If enabled, does not remove previous builds')
def build(no_clean):
    """Builds sdist and bdist wheel"""
    if not no_clean:
        status = _remove_build_dir()
        if status != 0:
            return status

    return _build_package()


@cli.command()
def clean():
    """Removes all build artifacts"""
    return _remove_build_dir()


@cli.command()
def cython():
    shell_run("cython ")


@cli.command()
@click.option('-u', '--user', 'user', default=None,
              help='PYPI username. Defaults to value in .pypirc if available')
@click.option('-p', '--password', 'password', default=None,
              help='PYPI password. Defaults to value in .pypirc if available')
@click.option('-b', '--build', 'build_', is_flag=True, default=False, help='If enabled, rebuilds package')
@click.option('--verbose', 'verbose', is_flag=True, default=True, help='Verbose output')
def upload(user, password, build_, verbose):
    """Uploads built package to PYPI"""
    cmd = [f"twine upload"]

    if build_:
        if _build_package() != 0:
            return 1

    pypirc = Path.home().joinpath('.pypirc')
    config = pypirc.read_text().splitlines() if pypirc.exists() else []
    if user is None:
        for line in config:
            if line.lower().startswith('username'):
                user = line.split()[-1].strip()
                break
        else:
            user = click.prompt("Enter your username", default='DannieBee', type=str)

    if password is None:
        for line in config:
            if line.lower().startswith('password'):
                password = line.split()[-1].strip()
                break
        else:
            password = click.prompt("Enter your password", hide_input=True, type=str)

    cmd.append(f'--user {user} --password {password}')
    if verbose:
        cmd.append('--verbose')

    cmd.append('dist/*')

    echo(f'Uploading packages to PyPI')
    shell_run(*cmd, tty=True)

    return 0


def _build_package():
    shell_run("make dist", tty=True)
    return 0


def _remove_build_dir():
    folder1 = ROOT.joinpath('build')
    folder2 = ROOT.joinpath('dist')
    try:
        if any([folder1.exists(), folder2.exists()]):
            echo('Removing previous package build folder artifacts.. ', nl=False)
            for f in (folder1, folder2):
                if f.exists():
                    shutil.rmtree(f)

            echo(style('Successful', 'green'))

        return 0
    except FileNotFoundError as e:
        echo(str(e), err=True)
        return 1
