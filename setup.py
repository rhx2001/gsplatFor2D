import glob
import os
import os.path as osp
import platform
import sys
import shutil

from setuptools import find_packages, setup

__version__ = None
exec(open("gsplat/version.py", "r").read())

URL = "https://github.com/nerfstudio-project/gsplat"

BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"
LINE_INFO = os.getenv("LINE_INFO", "0") == "1"

# 根据平台生成合适的 torch 要求
def get_torch_requirement():
    # 检查是否在虚拟环境中
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # 在虚拟环境中，使用宽松版本
        return "torch>=2.0.0"
    
    # 根据平台返回合适的版本字符串
    system = platform.system()
    arch = platform.machine()
    
    # 你可以根据平台返回不同的要求
    # 但通常建议使用宽松版本，让 pip 自动解决
    return "torch>=2.0.0"

def _ensure_cuda_env():
    # 如果已设置就别覆盖；但为了调试可打印出来
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        print(f"[gsplat] Current CUDA_HOME: {os.environ.get('CUDA_HOME')}")
        print(f"[gsplat] Current CUDA_PATH: {os.environ.get('CUDA_PATH')}")
        nvcc = shutil.which("nvcc")
        print(f"[gsplat] nvcc location: {nvcc}")
        return

    nvcc = shutil.which("nvcc")
    if nvcc:
        cuda_root = osp.abspath(osp.join(osp.dirname(nvcc), ".."))
        os.environ["CUDA_HOME"] = cuda_root
        os.environ.setdefault("CUDA_PATH", cuda_root)
        print(f"[gsplat] Detected CUDA at {cuda_root}")
        return

    win_candidates = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
    ]
    for c in win_candidates:
        if osp.isdir(c):
            os.environ["CUDA_HOME"] = c
            os.environ.setdefault("CUDA_PATH", c)
            print(f"[gsplat] Using fallback CUDA at {c}")
            return

    for c in ["/usr/local/cuda", "/opt/cuda"]:
        if osp.isdir(c):
            os.environ["CUDA_HOME"] = c
            os.environ.setdefault("CUDA_PATH", c)
            print(f"[gsplat] Using fallback CUDA at {c}")
            return


# <<< 关键：一开始就确保环境 >>>
_ensure_cuda_env()


def _patch_cpp_extension_cuda_home():
    """覆盖 torch.utils.cpp_extension 的全局 CUDA_HOME 缓存"""
    import torch.utils.cpp_extension as cpp_ext  # 注意：模块方式导入

    # 以环境变量为准；若还没有，就尝试从 nvcc 推断一次
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        nvcc = shutil.which("nvcc")
        if nvcc:
            cuda_home = osp.abspath(osp.join(osp.dirname(nvcc), ".."))
            os.environ["CUDA_HOME"] = cuda_home
            os.environ.setdefault("CUDA_PATH", cuda_home)

    # 显式覆盖模块内的全局变量（避免早期导入时缓存了 None）
    if cuda_home:
        try:
            cpp_ext.CUDA_HOME = cuda_home  # 关键行
            print(f"[gsplat] Patched cpp_extension.CUDA_HOME = {cuda_home}")
        except Exception as e:
            print(f"[gsplat] Failed to patch CUDA_HOME: {e}")


def get_ext():
    # <<< 在任何使用 cpp_extension 之前先打补丁 >>>
    _patch_cpp_extension_cuda_home()
    import torch.utils.cpp_extension as cpp_ext

    BuildExtension = cpp_ext.BuildExtension
    return BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)


def get_extensions():
    # <<< 同样先打补丁，再导入 CUDAExtension >>>
    _patch_cpp_extension_cuda_home()
    import torch
    import torch.utils.cpp_extension as cpp_ext

    CUDAExtension = cpp_ext.CUDAExtension

    extensions_dir = osp.join("gsplat", "cuda", "csrc")
    sources = glob.glob(osp.join(extensions_dir, "*.cu")) + glob.glob(
        osp.join(extensions_dir, "*.cpp")
    )
    sources = [p for p in sources if "hip" not in p]

    undef_macros = []
    define_macros = []

    if sys.platform == "win32":
        define_macros += [("gsplat_EXPORTS", None)]

    extra_compile_args = {"cxx": ["-O3"]}
    if os.name != "nt":
        extra_compile_args["cxx"] += ["-Wno-sign-compare"]

    # Windows 下不要传 -s
    if os.name == "nt":
        extra_link_args = []
    else:
        extra_link_args = [] if WITH_SYMBOLS else ["-s"]

    from torch.__config__ import parallel_info

    info = parallel_info()
    if (
        "backend: OpenMP" in info
        and "OpenMP not found" not in info
        and sys.platform != "darwin"
    ):
        extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
        if sys.platform == "win32":
            extra_compile_args["cxx"] += ["/openmp"]
        else:
            extra_compile_args["cxx"] += ["-fopenmp"]
    else:
        print("Compiling without OpenMP...")

    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_compile_args["cxx"] += ["-arch", "arm64"]
        extra_link_args += ["-arch", "arm64"]

    nvcc_flags = os.getenv("NVCC_FLAGS", "")
    nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
    nvcc_flags += ["-O3", "--use_fast_math"]
    if LINE_INFO:
        nvcc_flags += ["-lineinfo"]
    if torch.version.hip:
        define_macros += [("USE_ROCM", None)]
        undef_macros += ["__HIP_NO_HALF_CONVERSIONS__"]
    else:
        nvcc_flags += ["--expt-relaxed-constexpr"]
    
    host_cxx_flags = []
    if os.name == "nt":
        host_cxx_flags += ["/std:c++17", "/permissive-", "/Zc:__cplusplus"]
        # 可选：减少奇怪的宏干扰（例如 Windows 头里的 min/max）：
        host_cxx_flags += ["/DWIN32_LEAN_AND_MEAN", "/DNOMINMAX"]

    # 把这些 host 标志传给 NVCC
    for f in host_cxx_flags:
        nvcc_flags += ["-Xcompiler", f]

    extra_compile_args["nvcc"] = nvcc_flags
    if sys.platform == "win32":
        extra_compile_args["nvcc"] += ["-DWIN32_LEAN_AND_MEAN"]

    extension = CUDAExtension(
        "gsplat.csrc",
        sources,
        include_dirs=[osp.join(extensions_dir, "third_party", "glm")],
        define_macros=define_macros,
        undef_macros=undef_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    return [extension]


setup(
    name="gsplat",
    version=__version__,
    description=" Python package for differentiable rasterization of gaussians",
    keywords="gaussian, splatting, cuda",
    url=URL,
    python_requires=">=3.7",
    install_requires=[
        "jaxtyping",
        "rich>=12",
        get_torch_requirement(),
        # 你原来这里写了 'typing_extensions;' 会解析出错，修正：
        'typing_extensions; python_version<"3.8"',
    ],
    extras_require={
        "dev": [
            "black[jupyter]==22.3.0",
            "isort==5.10.1",
            "pylint==2.13.4",
            "pytest==7.1.2",
            "pytest-xdist==2.5.0",
            "typeguard>=2.13.3",
            "pyyaml==6.0",
            "build",
            "twine",
            "ninja",
        ],
    },
    ext_modules=get_extensions() if not BUILD_NO_CUDA else [],
    cmdclass={"build_ext": get_ext()} if not BUILD_NO_CUDA else {},
    packages=find_packages(),
    include_package_data=True,
)
