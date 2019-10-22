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
      install_requires=['numpy==1.16.4','tensorflow==1.14.0','tensorflow-datasets==1.2.0','tensorflow-hub==0.5.0','decorator==4.4.0','cloudpickle==1.2.1','tensorboard'],
      )
