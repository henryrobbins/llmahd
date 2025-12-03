import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="llamda",
    version="0.0.1",
    author="Henry Robbins",
    author_email="hw.robbins@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.10, <3.12",
    install_requires=[
        "Jinja2",
        "litellm",
        "openai",
        "numpy",
        "torch",
        "scipy",
        "scikit-learn",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3",
            "black>=25.1",
            "flake8>=7.1",
            "mypy>=1.16",
            "pytest-cov>=6.0",
            "sphinx>=8.1",
            "sphinx-rtd-theme>=3.0",
        ]
    },
    package_data={"llamda": ["prompts/*", "problems/*"]},
)
