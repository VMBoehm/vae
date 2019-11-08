from setuptools import setup

setup(name='vae',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description='a vanilla variational autoencoder',
      url='http://github.com/VMBoehm/vae',
      author='Vanessa Martina Boehm',
      author_email='vboehm@berkeley.edu',
      license='Apache License 2.0',
      packages=['vae'],
      )
