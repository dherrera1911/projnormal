from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='projected_normal',
    url='https://github.com/dherrera1911/projected_normal_distribution',
    author='Daniel Herrera-Esposito',
    author_email='dherrera1911@gmail.com',
    # Needed to actually package something
    packages=['projected_normal'],
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
    description='Python implementation of projected normal distribution functionalities.',
    # long_description=open('README.txt').read(),
)
