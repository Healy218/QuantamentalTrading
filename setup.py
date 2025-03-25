from setuptools import setup, find_packages

setup(
    name="quantamental_trading",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'scipy',
        'streamlit',
        'tqdm'
    ],
) 