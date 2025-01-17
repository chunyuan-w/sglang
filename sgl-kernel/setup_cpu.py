from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

cxx_flags = ["-O3"]
libraries = ["c10", "torch", "torch_python"]
extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", "-L/usr/lib/x86_64-linux-gnu"]
ext_modules = [
    CppExtension(
        name="sgl_kernel_cpu",  # Python module name
        sources=["src/sgl-kernel/csrc/cpu/all_reduce.cpp"],
        extra_compile_args={
            "cxx": cxx_flags,
        },
        libraries=libraries,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="sgl-kernel-cpu",
    packages=["sgl_kernel-cpu"],
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
