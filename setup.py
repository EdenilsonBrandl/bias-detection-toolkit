from setuptools import setup, find_packages

setup(
    name='bias_detection_toolkit',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy'
    ],
    author='Edenilson Brandl',
    author_email='engbrandl@yahoo.com.br',
    description='A toolkit to detect bias and anomalies in datasets used in machine learning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/bias-detection-toolkit',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)