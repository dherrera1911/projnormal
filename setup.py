from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='quadratic_ratios',
    url='https://github.com/dherrera1911/quadratic_ratios',
    author='Daniel Herrera-Esposito',
    author_email='dherrera1911@gmail.com',
    # Needed to actually package something
    packages=['qr_library'],
    # Needed for dependencies
    install_requires=[
      'numpy',
      'torch',
      'matplotlib',
      'scipy'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='Python implementation of quadratic ratios statistics',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
