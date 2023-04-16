from setuptools import find_packages, setup

requirements = [
    "tensorflow",
    "numpy",
    "scipy",
    "matplotlib",
    "pandas"
]
setup(
    name="TrafficPrediction",
    version="0.0.1",
    description="""
    Traffic Prediction
    """,
    # long_description=open('../README.rst').read(),
    # long_description_content_type="text/markdown",
    author="TrafficPrediction Team",
    # author_email='gaurav.singh.1917@student.uu.se',
    python_requires=">=3.7",
    license="Apache License 2.0",
    zip_safe=False,
    # entry_points={
    #     'console_scripts': [""]
    # },
    install_requires=requirements,
    keywords="LSTM forecasting",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)