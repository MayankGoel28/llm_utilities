from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name='llm_utilities',
    version='0.1',
    packages=find_packages(),
    install_requires=install_requires,
    author='Mayank Goel',
    description='Utilities for working with LLMs',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/MayankGoel28/llm_utilities',
)