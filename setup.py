from setuptools import setup, find_packages

with open('requirements.txt') as f:
    reqs = f.read()

reqs = reqs.strip().split('\n')

install = [req for req in reqs if not req.startswith("git+git://")]
depends = [
    req.replace("git+git://", "git+http://") for req in reqs
    if req.startswith("git+git://")
]

setup(
    name='vitaminc',
    python_requires='>=3.7',
    packages=find_packages(),
    install_requires=install,
    dependency_links=depends,
)
