from setuptools import setup, find_packages

setup(
    name="socia_camel",
    version="0.1.0",
    description="基于CAMEL架构的社会模拟智能体系统",
    author="SOCIA-CAMEL Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "networkx>=2.6.0",
        "pyyaml>=6.0",
        "dependency-injector>=4.40.0",
        "openai>=1.0.0",
        "google-generativeai>=0.3.0",
        "anthropic>=0.5.0",
        "llama-cpp-python>=0.2.0",
        "requests>=2.28.0",
        "tqdm>=4.64.0"
    ],
    python_requires=">=3.8",
) 