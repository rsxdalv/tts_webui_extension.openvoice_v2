import setuptools

setuptools.setup(
    name="tts_webui_extension.openvoice_v2",
    packages=setuptools.find_namespace_packages(),
    version="0.1.0",
    author="rsxdalv",
    description="OpenVoice: A versatile instant voice cloning approach",
    url="https://github.com/rsxdalv/tts_webui_extension.openvoice_v2",
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

