from setuptools import setup, find_packages

setup(
   name='mjcf2o3d',
   version='0.0.0',
   description='Scans Mujoco models as point clouds',
   author='Charlie Gauthier',
   author_email='charlie-gauthier@outlook.com',
    packages=find_packages(),
    install_requires=["typing-extensions"]
)


