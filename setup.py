# setup.py
from setuptools import setup, find_packages

setup(
    name="voxcpm",
    version="1.0.4",
    packages=find_packages(where="src"),          # 👈 自动发现所有包
    package_dir={"": "src"},                      # 👈 告诉 setuptools 包在 src/ 下
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.5.0",
        "torchaudio>=2.5.0",
        "transformers>=4.36.2",
        "einops",
        "gradio",
        "inflect",
        "addict",
        "wetext",
        "modelscope>=1.22.0",
        "datasets>=3,<4",
        "huggingface-hub",
        "pydantic",
        "tqdm",
        "simplejson",
        "sortedcontainers",
        "soundfile",
        "funasr",
        "spaces"
    ],
    entry_points={
        "console_scripts": [
            "voxcpm=voxcpm.cli:main",
        ],
    },
    author="OpenBMB",
    author_email="openbmb@gmail.com",
    description="VoxCPM: Tokenizer-Free TTS for Context-Aware Speech Generation and True-to-Life Voice Cloning",
    license="Apache-2.0",
    keywords=["voxcpm", "text-to-speech", "tts", "speech-synthesis", "voice-cloning", "ai", "deep-learning", "pytorch"],
    url="https://github.com/OpenBMB/VoxCPM",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)