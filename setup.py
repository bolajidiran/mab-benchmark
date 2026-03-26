from setuptools import setup, find_packages

setup(
    name="mab_benchmark",
    version="1.0.0",
    description=(
        "MAB Unified Benchmark Suite — Gap 5 Research Programme. "
        "Five settings, four statistical requirements, BCS bridge proxy, "
        "Papers With Code leaderboard."
    ),
    author="Gap 5 Research Programme",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=2.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
