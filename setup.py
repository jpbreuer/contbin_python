from setuptools import setup, find_packages

setup(
    name='pycontbin',
    version='0.2.1',    
    description='A Python implementation of the ContBin algorithm',
	long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/jpbreuer/contbin_python',
    author='Jean-Paul Breuer',
    author_email='jeanpaul.breuer@gmail.com',    
    license='BSD 2-clause',
    packages=['pycontbin'],
    install_requires=['numpy',
					  'astropy',
					  'pyds9',
                      ],

    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',        
    ],
	python_requires='>=3.6'
)   