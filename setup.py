import setuptools

setuptools.setup(
    name="extension_openvoice_v2",
    packages=setuptools.find_namespace_packages(),
    version="0.0.2",
    author="rsxdalv",
    description="OpenVoice: A versatile instant voice cloning approach",
    url="https://github.com/rsxdalv/extension_openvoice_v2",
    project_urls={},
    scripts=[],
    install_requires=[
        "MyShell-OpenVoice @ git+https://github.com/rsxdalv/OpenVoice@stable",
        "melotts @ git+https://github.com/rsxdalv/MeloTTS@stable",
        # "langid",
        # "numpy",
        # "torch>=2.0.0",
        # "soundfile",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
