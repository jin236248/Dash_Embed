from setuptools import setup, find_packages

setup(
    name='dash-word-embedding',
    version='0.1',
    packages=find_packages(where='app'),
    package_dir={'': 'app'},
    install_requires=[
        'dash',
        'plotly',
        'numpy',
        'pandas',
        'scikit-learn',
        'json'
    ],
    entry_points={
        'console_scripts': [
            'run-app=app.app:run_server',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)