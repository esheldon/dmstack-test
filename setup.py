import glob
from setuptools import setup, find_packages

scripts = glob.glob('bin/*')
scripts = [s for s in scripts if '~' not in s]

setup(
    name="dmstack_test",
    version="0.1.0",
    packages=find_packages(),
    scripts=scripts,
    author='Erin Sheldon',
    author_email='erin.sheldon@gmail.com',
    url='https://github.com/esheldon/dmstack-test',
)
