from setuptools import setup, find_packages
!pip install google-generativeai

setup(
    name="auto-service-assistant",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.31.0",
        "openai>=1.11.0",
        "google-generativeai>=0.3.2",
        "pandas>=2.1.4",
        "pillow>=10.1.0",
        "aiohttp>=3.9.1",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "asyncio>=3.4.3",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A multi-agent automotive service system",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/auto-service-assistant",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
