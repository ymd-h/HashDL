from setuptools import Extension, setup, find_packages


rb_source = "HashDL/hashdl"
cpp_ext = ".cpp"
pyx_ext = ".pyx"

setup_requires = ["numpy"]

if platform.system() == 'Windows':
    extra_compile_args = ["/std:c++20", "/O2"]
    extra_link_args = None
    if debug:
        extra_compile_args.append('/DCYTHON_TRACE_NOGIL=1')
else:
    extra_compile_args = ["-O3","-std=c++20","-march=native"]
    extra_link_args = ["-std=c++20", "-ltbb"]
    if debug:
        extra_compile_args.append('-DCYTHON_TRACE_NOGIL=1')

# Check cythonize or not
cpp_file = rb_source + cpp_ext
pyx_file = rb_source + pyx_ext
use_cython = (not os.path.exists(cpp_file)
              or (os.path.exists(pyx_file)
                  and (os.path.getmtime(cpp_file) < os.path.getmtime(pyx_file))))
if use_cython:
    suffix = pyx_ext
    setup_requires.extend(["cython>=0.29"])
    compiler_directives = {'language_level': "3"}

    if debug:
        compiler_directives['linetrace'] = True
else:
    suffix = cpp_ext


# Set ext_module
ext = [["HashDL","hashdl"]]

ext_modules = [Extension(".".join(e),
                         sources=["/".join(e) + suffix],
                         extra_compile_args=extra_compile_args,
                         extra_link_args=extra_link_args,
                         language="c++") for e in ext]

class LazyImportBuildExtCommand(build_ext):
    """
    build_ext command class for lazy numpy and cython import
    """
    def run(self):
        import numpy as np

        self.include_dirs.append(np.get_include())
        build_ext.run(self)

    def finalize_options(self):
        if use_cython:
            from Cython.Build import cythonize
            self.distribution.ext_modules = cythonize(self.distribution.ext_modules,
                                                      compiler_directives=compiler_directives,
                                                      include_path=["."],
                                                      annotate=True)
        super().finalize_options()


setup(name="HashDL",
      version="0.0.0",
      install_requires=["numpy"],
      setup_requires=setup_requires)
