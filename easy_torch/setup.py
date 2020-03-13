
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='easy-torch',  
     version='0.1',
     scripts=['easy-torch'] ,
     author="Jean-Baptiste Remy",
     author_email="remyjeanb@gmail.com",
     description="A package for easy prototyping and management of Pytorch models",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/JbRemy/easy-torch",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
