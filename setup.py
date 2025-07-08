from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Get the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Read the README for long description
with open(os.path.join(base_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="nio-jma",
    version="0.1.0",
    description="Neural Inverse Operators",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Jonathan Ma",
    author_email="johnma@udel.edu",
    packages=find_packages(include=['core', 'core.*', 'utils', 'utils.*', 'training', 'training.*', 'data', 'data.*']),
    namespace_packages=['core', 'utils', 'training', 'data'],
    install_requires=requirements,
    python_requires='>=3.7',
    # Include package data files
    include_package_data=True,
    # Add any additional package data
    package_data={
        '': ['*.yaml', '*.json', '*.h5', '*.pt', '*.pth', '*.md'],
    },
    # Add entry points
    entry_points={
        'console_scripts': [
            'nio-train=training.scripts.RunNio:main',
        ],
    },
    # Add project URLs
    project_urls={
        'Source': 'https://github.com/jma02/nio-jma',
    },
    # Add classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
