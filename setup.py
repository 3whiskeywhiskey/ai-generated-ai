from setuptools import setup, find_packages

def get_requirements():
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
    return requirements

setup(
    name="llm_project",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=get_requirements(),
    python_requires='>=3.8',
    author="Your Name",
    author_email="your.email@example.com",
    description="A tensor-parallel language model implementation",
    keywords="deep-learning, nlp, language-models, tensor-parallelism",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ],
) 