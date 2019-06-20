from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[
             Extension("time_integrator",
                       ["time_integrator.pyx"],
                       libraries=["m"],
                       extra_compile_args = ["-O3", "-ffast-math", "-march=native" ]
                    
                       )
             ]

setup(
      name = "time_integrator",
      cmdclass = {"build_ext": build_ext},
      ext_modules = cythonize(ext_modules)
      )

ext_modules=[
            Extension("space_dicretizator",
                      ["space_dicretizator.pyx"],
                      libraries=["m"],
                      extra_compile_args = ["-O3", "-ffast-math", "-march=native" ]
                      
                      )
            ]

setup(
      name = "space_dicretizator",
      cmdclass = {"build_ext": build_ext},
      ext_modules = cythonize(ext_modules)
      )
