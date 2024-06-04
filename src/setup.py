import setuptools

requirements = (
    "numpy>=1.23.5"
)

extra_requirements = {
    "tests": (
        "pytest>=7.4.4"
    )
}

setuptools.setup(
    name="ptree_gp",
    version="0.0.1",
    author="Eremeev Dmitrii",
    author_email="eremeev.dima.2002@mail.ru",
    description="Gaussian processes on phylogenetic trees",
    packages=["ptree_gp"],
    install_requires=requirements,
    extras_require=extra_requirements
)