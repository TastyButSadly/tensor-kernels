from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="moe-topk",
    version="0.1.0",
    description="Stable CPU top-k expert selection for PyTorch",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    # Keep the build and runtime C++ APIs on the same PyTorch release.
    install_requires=["torch==2.8.0"],
    ext_modules=[
        CppExtension(
            name="moe_ops._C",
            sources=["src/moe_ops/ops.cpp"],
            extra_compile_args=["-O3", "-g0"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
