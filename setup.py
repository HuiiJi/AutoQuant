from setuptools import setup, find_packages

setup(
    name="autoquant",
    version="1.0.0",
    description="专业的AI模型量化工具链条，参考MQBench设计，支持PTQ/QAT、混合精度量化、符号化追踪和ONNX导出",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "onnx>=1.14.0",
        "numpy>=1.23.0",
        "tabulate>=0.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "onnxruntime>=1.14.0",
            "onnxsim>=0.4.0",
            "matplotlib>=3.0.0",
        ],
        "full": [
            "onnxsim>=0.4.0",
            "matplotlib>=3.0.0",
            "pandas>=2.0.0",
        ]
    },
    # 这是关键！注册命令行工具
    # entry_points的作用：让pip安装后可以直接用 'autoquant' 命令
    entry_points={
        "console_scripts": [
            # 格式：命令名=模块:函数
            "autoquant=autoquant.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
)