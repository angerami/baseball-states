from setuptools import setup

setup(
    name="baseball-states",
    version="0.1.0",
    packages=["baseball_states"],
    package_dir={"baseball_states": "src"},
    install_requires=open('requirements.txt').read().splitlines(),
)