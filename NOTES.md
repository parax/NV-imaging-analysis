# Table of Contents


<!-- MarkdownTOC -->

- [To-Do](#to-do)
  - [In order of Priority:](#in-order-of-priority)
  - [Troubleshooting](#troubleshooting)
- [Speedups](#speedups)
  - [Plan](#plan)
  - [Future choices for Fitting speedup](#future-choices-for-fitting-speedup)
    - [GPUfit](#gpufit)
    - [Julia](#julia)
      - [Fitting in Julia](#fitting-in-julia)
      - [GPU integration in Julia](#gpu-integration-in-julia)
      - [Parallelisation in Julia](#parallelisation-in-julia)

<!-- /MarkdownTOC -->



# To-Do



## In order of Priority:


- restructure code to follow best practices. Rename repo to something without underscores, and structure with a setup file and (e.g.) \_\_init\_\_.py files as in this [link](https://python-packaging.readthedocs.io/en/latest/minimal.html). Make it outwardly presentable. 'Future' possibility is adding it to conda-forge and/or PyPi.
  - This is necessary right now actually. To import any files that aren't children requires the program to be structured as if it were a module. This is necessary for the test cases.
  - More help [here](https://packaging.python.org/)
- Proposed structure:

```
process_widefield/
+-- process_widefield/
|   +-- __init__.py
|   +-- process_raw/
|   |   +-- __init__.py
|   |   +-- {py files}
|   +-- reconstruction/
|   |   +-- __init__.py
|   |   +-- {py files}
|   +-- {etc. more py files}
|   +-- tests/
|   |   +-- __init__.py
|   |   +-- test_class.py
|   |   +-- profiler.py
|   |   +-- {cases}
|   +-- scripts/
|   |   +-- profiler.py
|   |   +-- test.cmd
|   |   +-- test.sh
|   |   +-- run_processor(.py/.md) {new main file}
|   +-- docs/
|   |   +-- {documentation}
|   +-- data/
|   |   +-- {test_data etc}
|   +-- options/
|   |   +-- options.json
|   |   +-- {etc}
+-- setup.py
+-- README.md
+-- NOTES.md
+-- manifest.ini
```


- [ ] Find best value for chunksize

- [ ] Check jacobian speed etc. on single sweep test data

- [ ] add comments to json files

- [ ] add residual below plots of fits (test on single sweep first)

- [ ] Really need some documentation of the options.json file, as well as some documentation of 'often used' option files (perhaps referring to test case json files)

- [ ] Ignore ref not working (reshape array error, tested on singlepeak dataset)

- [ ] Add pytest assertions to code (perhaps at just the lowest stack level) and add documentation for it

- [ ] Delete some of the unused branches on GitLab for cleanliness

- [ ] Write more jacobians and determine whether they're as accurate as numeric results (NB automatic diffs in Julia = Fast) -- then _really_ need to speedtest this to see if we get a worthwile speedup (we should)

- [ ] Add an object heirachy flowchart & data flow flowchart

- [ ] error checkers in more fns (add docstring at the same time)

- [ ] idiot proof parameters
  - specifically in going ahead with parallelpool etc. below

- [ ] add 'proceed to parallel process' query (or add option to skip it)
  - then close plts (or open all at end if skipped)

- [ ] template options files - these should flow nicely from the test case options

- [ ] file dialog for selecting the processing file wanted

- [ ] !!! never override a json file on save - add an idx etc.

- [ ] !!! graph residual below fit for full_roi

- [ ] Add auto devops, add CI/CD

- [ ] When reading options make sure num peaks matches number of fitting inputs (write something like this for all options)

- [ ] Better environment handler than conda (maybe pyenv or something)

- [ ] Really need to clean up where the options are processed, for example the fit options are all over the place (this is my (Sam's) fault)


## Troubleshooting


- If the program hangs, and you have low binning, try increasing the chunksize sent to the parallel tool.



# Speedups


## Plan


Primarily: try Numba, different parallelising tools. As well as streamlining the actual fitting process. This could possibly be sped up by trying a different fitting function/passing into a different language such as julia.

I really think we should try out some different compilers as well. Speed testing [Numba](http://numba.pydata.org/), [Cython](https://cython.org/) and [PyPy](https://pypy.org/) will give us a good idea of the possibilities. Not to mention Numba in particular has built in GPU Acceleration, parallel threading etc. Compare all in new speedtesting suite.

[Limits](https://medium.com/coding-with-clarity/speeding-up-python-and-numpy-c-ing-the-way-3b9658ed78f4) of Numpy - perhaps use Numba @jit(nopython=True) to speed up the function calls.

DB also mentioned a graph Sam found of different parallelising tools and their speedups for different sized datasets, Sam can't find it but this would be benefitial. - At least understand the different approaches and their benefits etc.


## Future choices for Fitting speedup


### GPUfit


NB: Problem with GPUfit: we need to compile the fitmodel before running - difficult in Windows, but also not as flexible as we wish.


First attempt should be to use this nifty little GPU accelerated lmfit package [GPUfit](https://github.com/gpufit/Gpufit), which has a Python [binding](https://gpufit.readthedocs.io/en/latest/bindings.html#python).

What I have been able to work out thus far is that it grabs _all_ of the data/fits/etc. we want to run at the function call, then sends it all over to the GPU in chunks (depending on available VRAM) and does the fitting in parallel and vectorised. To speed this up we will want all fit functions we send over to it to be as fast as possible. Perhaps using Cython or Numba on the bottleneck parts of our code.

It is written for C, and has more customization options over there (for example, Fit Models akin to our own). It does have disadvantages though, as stated in the docs:
> A current disadvantage of the Gpufit library, when compared with established CPU-based curve fitting packages, is that in order to add or modify a fit model function or a fit estimator, the library must be recompiled. We anticipate that this limitation can be overcome in future releases of the library, by employing run-time compilation of the CUDA code.

If the Python wrapper is not enough for us we could always do the fitting in C/Cython/CPython.


### Julia


Probably not our first port of call, this would require a somewhat more involved restructing or the code. Only the fitting would need to be undertaken in Julia (i.e. pass data from Python to Julia then return the results and continue as before), yet I think it's easier to keep our current fitmodel OO style in Python for the timebeing - at least until we determine where our bottlenecks are (see Plan above). It would be fun to play around with it though!


- Julia generates native machine code directly before a function is first run (Just In Time, JIT)
- Performance approaches that of C
- Designed for parallel and distributed computing (and supports CUDA)
- Can use ccall keyword to call C-exported or Fortran shared library functions (individually and *directly*), or PyCall to call python functions
- Dynamically typed, and user defined types are just as fast as built-in
- Mutliple dispatch (define function behaviour differently depending on combinations of argument type)


Note that to get it to C speeds you need to specify types [link](https://sylvaticus.gitbooks.io/julia-language-a-concise-tutorial/content/language-core/performances.html), this is equivalent to [numba](https://numba.pydata.org/) in Python. The first link there also describes how to profile in Julia.

Calling [Python](https://sylvaticus.gitbooks.io/julia-language-a-concise-tutorial/content/language-core/interfacing-julia-with-other-languages.html) and calling from Python [pyjulia](https://github.com/JuliaPy/pyjulia)


#### Fitting in Julia


- [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/), a package written completely in Julia that provides many of its benefits (automatic differentiation through [JuliaDiff](http://www.juliadiff.org/), multiple dispatch etc.)
- [LsqFit.jl](https://github.com/JuliaNLSolvers/LsqFit.jl), a simple package derived from the above, uses lmfit alg.
- [NLopt.jl](https://github.com/JuliaOpt/NLopt.jl) interface to the [NLopt](https://nlopt.readthedocs.io/en/latest/NLopt_Introduction/) free/open-source NLopt library for nonlinear optimization. Quote from manual:
> However, our experience is that, for nonlinear optimization, the best algorithm is highly problem-dependent; the best approach is to try several different techniques and see which one works best for you. NLopt makes this easy by providing a single package and a common interface for many different algorithms, callable from many different languages.
- [BlackBoxOptim.jl](https://github.com/robertfeldt/BlackBoxOptim.jl) global minimum provides parallelization (for each optimisation call, i.e. if each fit takes a long time we could use that - only really an option if we can parallelise over multiple PCs)
- [CurveFit.jl](https://github.com/pjabardo/CurveFit.jl) simple least squares (nonlinear included) fitting functions, doesn't look as fancy as the above packages.


#### GPU integration in Julia


[ArrayFire.jl](https://github.com/JuliaGPU/ArrayFire.jl), Julia interface to ArrayFire. CUDA/OpenCL backends, built in image processing, FFT's, vector ops/linalg etc. but no fitting tools

[CUDAdrv.jl](https://github.com/JuliaGPU/CUDAdrv.jl), direct CUDA driver


#### Parallelisation in Julia


Parallel Processing is built into Julia, see this [link](https://nbviewer.jupyter.org/github/sylvaticus/juliatutorial/blob/master/assets/Parallel%20computing.ipynb), looks easy but I'm sure it's more difficult than it seems.


More complicated discussion [here](https://docs.julialang.org/en/v1/manual/parallel-computing/index.html), in particular [Shared Arrays](https://docs.julialang.org/en/v1/manual/parallel-computing/index.html#man-shared-arrays-1)
