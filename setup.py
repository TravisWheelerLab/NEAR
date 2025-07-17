from setuptools import setup, Extension, find_packages, Command
from setuptools.command.build_py import build_py as _build_py
import os
import subprocess
import shutil
import sys

# ─────────────────────────────────────────────────────────────────────────────
# 1) C extension: we give it a distinct name (e.g. "_process_near") so it does
#    not collide with the "process_near_results" executable.  When you import
#    it in Python, you would do something like:
#
#        from near._process_near import <function_or_symbol>
#
#    rather than "near.process_near_results", which is what the old script did.
# ─────────────────────────────────────────────────────────────────────────────

debug_args = ["-O1", "-DDEBUG=1", "-fsanitize=address", "-fno-omit-frame-pointer", "-fno-optimize-sibling-calls"]
normal_args = ['-O3', '-Wall', '-Wextra', '-Wuninitialized', '-Wmaybe-uninitialized', '-Wstrict-overflow=5']
c_extension = Extension(
    name="near._process_near",
    sources=[
        "src/c/main.c",
        "src/c/types.c",
        "src/c/io.c",
        "src/c/process_hits.c",
        "src/c/util.c",
    ],
    include_dirs=["src/c"],
    extra_compile_args=normal_args,
    extra_link_args=["-lm"],
)

# ─────────────────────────────────────────────────────────────────────────────
# 2) Custom Command to compile the standalone "process_near_results" executable
#    before any Python files are copied/installed.  We subclass setuptools.Command
#    (not distutils) so that everything hooks into setuptools’s build order.
# ─────────────────────────────────────────────────────────────────────────────
class BuildExecutable(Command):
    description = "compile the process_near_results C program into a standalone binary"
    user_options = []  # no command-line options.

    def initialize_options(self):
        # (no options to initialize)
        pass

    def finalize_options(self):
        # (nothing to finalize)
        pass

    def run(self):
        # 2a) ensure the build/exe directory exists
        build_exe_dir = os.path.join("build", "exe")
        os.makedirs(build_exe_dir, exist_ok=True)

        # 2b) pick up CC from environment or default to gcc
        cc = os.environ.get("CC", "gcc")

        # 2c) compile all sources into one binary
        c_sources = [
            "src/c/main.c",
            "src/c/types.c",
            "src/c/io.c",
            "src/c/process_hits.c",
            "src/c/util.c",
        ]
        # Put "-I src/c" near the front; "-lm" near the end.
        debug_args = ["-O0", "-g", "-DDEBUG=1", "-fsanitize=address", "-fno-omit-frame-pointer"]
        normal_args = ['-O3']
        compile_cmd = [cc] + normal_args + ["-I", "src/c"] + c_sources + ["-o", os.path.join(build_exe_dir, "process_near_results"), "-lm"]

        self.announce(f"Running: {' '.join(compile_cmd)}", level=3)
        subprocess.check_call(compile_cmd, shell=False)

        # 2d) copy the resulting executable into "src/near/bin/"
        target_dir = os.path.join("src", "near", "bin")
        os.makedirs(target_dir, exist_ok=True)

        src_exe = os.path.join(build_exe_dir, "process_near_results")
        dst_exe = os.path.join(target_dir, "process_near_results")

        shutil.copy2(src_exe, dst_exe)
        os.chmod(dst_exe, 0o755)
        self.announce(f"Copied executable to {dst_exe}", level=3)


# ─────────────────────────────────────────────────────────────────────────────
# 3) Command to copy models directory to package directory
# ─────────────────────────────────────────────────────────────────────────────
class CopyModels(Command):
    description = "copy models directory to the package directory"
    user_options = []  # no command-line options.

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Copy models directory to src/near/models
        src_models_dir = "models"
        dst_models_dir = os.path.join("src", "near", "models")

        # Ensure the destination directory exists
        os.makedirs(dst_models_dir, exist_ok=True)

        # Copy each model file
        for item in os.listdir(src_models_dir):
            src_path = os.path.join(src_models_dir, item)
            dst_path = os.path.join(dst_models_dir, item)

            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                self.announce(f"Copied model file {item} to {dst_path}", level=3)
            elif os.path.isdir(src_path):
                if os.path.exists(dst_path):
                    shutil.rmtree(dst_path)
                shutil.copytree(src_path, dst_path)
                self.announce(f"Copied model directory {item} to {dst_path}", level=3)


# ─────────────────────────────────────────────────────────────────────────────
# 4) Subclass build_py so that we run our BuildExecutable command
#    *before* setuptools copies any pure-Python modules into build/lib/.
# ─────────────────────────────────────────────────────────────────────────────
class build_py(_build_py):
    def run(self):
        # First: compile+copy the C executable
        self.run_command("build_executable")
        # Then do the normal build_py steps (copying .py files, etc.)
        self.run_command("copy_models")

        super().run()


# ─────────────────────────────────────────────────────────────────────────────
# 5) Finally, call setup().  We register both the extension and our new commands.
# ─────────────────────────────────────────────────────────────────────────────
setup(
    name="near",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),

    # 5a) C-extension module (accessible as near._process_near)
    ext_modules=[c_extension],

    # 5b) cmdclass: tie "build_py" to our subclass, and register custom commands
    cmdclass={
        "build_py": build_py,
        "build_executable": BuildExecutable,
        "copy_models": CopyModels,
    },

    # 5c) ensure that "near/bin/process_near_results" and models end up in the installed package
    package_data={
        "near": ["bin/process_near_results", "models/*"],
    },
    include_package_data=True,  # in case you have other data files

    # 5d) runtime dependencies
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
        "tqdm",
        "pyyaml",
    ],

    # 5e) if you want a console‐script entrypoint for your python code:
    entry_points={
        "console_scripts": [
            "near=near.main:main",
        ],
    },

    python_requires=">=3.11.11",
)
