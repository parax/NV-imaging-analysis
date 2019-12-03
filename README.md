# Table of Contents

<!-- MarkdownTOC -->

- [Design](#design)
- [Setup/Installation](#setupinstallation)
    - [Text Editor](#text-editor)
- [Version Control System](#version-control-system)
    - [Installation](#installation)
    - [Connection to PyCharm](#connection-to-pycharm)
- [Virtual Environment \(Conda\)](#virtual-environment-conda)
    - [Quickstart](#quickstart)
    - [Packages](#packages)
    - [Sharing/Importing Environment](#sharingimporting-environment)
- [Basics](#basics)
    - [Naming](#naming)
    - [Documentation](#documentation)
    - [Global Design](#global-design)

<!-- /MarkdownTOC -->


**Important** Read the NOTES.md file after this one.



# Design


This project holds a few different pieces all currently aimed at processing widefield nv imaging files. The global object heirachy is outlined at the bottom of this page.



# Setup/Installation


## Text Editor


The default and recommended text editor is [PyChalm](https://www.jetbrains.com/pycharm/). It is a fully featured Integrated Development Environment, much like RConsole or MATLAB. The recommended version is Professional (you should get a free version through a student account). PyChalm by default comes setup with Git and Conda integration, as well as linters/syntax checkers and everything else you could want plus some extras you didn't realise you wanted.

Select the 'Add launchers to PATH' and "create associations with .py" options in the setup wizard.

To make life simpler on you, don't open PyCharm until said so below.


# Version Control System


The project is housed on [Gitlab](https://gitlab.unimelb.edu.au/QSL/process-widefield-py), you will need to be given access by an owner and sign in with uni credentials. To communicate with the Gitlab server you will need to setup an ssh key (on each device connected, there will need to be one on each lab computer as well). My installation instructions below are taken from the Gitlab Docs [here](https://docs.gitlab.com/ee/ssh/).

You can also use Gitlab in-browser, i.e. not using the git commands at all. This is not recommended but can be great in a pinch.

Tip: It's a lot easier to read diffs/merges using the side-by-side visual. Somewhere in your diff tool/online it will have this options, give it a shot.


## Installation


If you're on windows you will need to download a Git/OpenSSH client (Unix systems have it pre-installed). The simplest way to do this for integration with PyCharm is just to use [Git for Windows](https://gitforwindows.org/), even if you have WSL/Cygwin. Just do it, it isn't that big a package.

Open the Git Bash terminal (or Bash on Unix). Generate a new ED25519 SSH key pair:

```Bash
ssh-keygen -t ed25519 -C "<YOUR UNI EMAIL HERE>"
```

You will be prompted to inpu a file path to save it to. Just click Enter to put it in the default \~/.ssh/config file
Once that's decided you will be prompted to input a password. To skip press Enter twice (make sure you don't do this for a shared PC such as the Lab computers).

Now add to your Gitlab account (under settings in your browser)
To clip the public key to your clipboard for pasting, in Git Bash:

```Bash
cat \~/.ssh/id_ed25519.pub | clip
```

macOS:

```Bash
pbcopy < \~/.ssh/id_ed25519.pub
```

WSL/GNU/Linux (you may need to install xclip but it will give you instructions):

```Bash
xclip -sel clip < \~/.ssh/id_ed25519.pub
```

Now we can check whether that worked correctly:

```Bash
ssh -T git@git.unimelb.edu.au
```

If it's the first time you connect to Gitlab via SSH, you will be asked to verify it's authenticity (respond yes). You may also need to provide your password. If it responds with *Welcome to Gitlab, @username!* then you're all setup.


## Connection to PyCharm


You now need to point PyCharm to this git file.  On windows, this is stored in (there are a few in the Git directory, for example Git\cmd\bin.exe, I'm rolling with this one as its the one returned from where git):

To find out which git you have configured in terminal, type (in windows Git Bash):

```Bash
where git
```

For me this was:

```Bash
C:\Program Files\Git\bin\git.exe
```

On Unix to find out where it's stored, type:

```Bash
which git
```

Copy this path.

Open PyCharm for the first time, and instead of creating a Project or anything like that, go into the bottom right hand corner and click on configure, go down to Version control, Git, and copy in the path to the git we copied above.

If you haven't pulled the repo down yet, you can use the checkout from VCS option. (haven't tried this but read the below just in case any similar options show up -- especially with the setup of the conda environment!). Note all of these things can be changed later, its just a very messy menubar/settings so I can't point you in the right directions ahead of time.

Else (assuming you have also *already* pulled the repo so have a conda environment set up as below i.e. *already* made an environment from the repo's yaml file), you can then start a new project in Pure Python. Give it a good name like pw_project. Click on the project interpreter button, existing interpreter, and specify the conda environment for the repo's yaml file.

Further settings of particular use:
- Go to Settings - Editor - TODO
    - add patterns for FIXME, README and NOTE (as well as TODO) i.e. \bFIXME\b.*
    - this will make life easier as these already exist in the scripts and in particular are highlighted for fixing before you commit. Beautiful!


# Virtual Environment (Conda)


Conda/Anaconda is a software package that allows for multiple 'environments' i.e. collections of packages, even different versions of Python, on the same machine. It also allows us to share our environment on our personal machines between people so we can all have a common 'background' to run the code on - very useful for avoiding errors!

Conda/Anaconda is a software package that allows for multiple 'environments' i.e. collections of packages, even different versions of Python, on the same machine. It also allows us to share our environment on our personal machines between people so we can all have a common 'background' to run the code on - very useful for avoiding errors!

There are various methods for installing conda, but PyCharm comes with it. If you like working from the command line, download [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
NB: If not using PyCharm, and using a gui for your gitting, ensure your git/gui has access to the environment you're using.

The Gitlab repo now has a specified conda environment YAML file. Best to use and update it so everyone can run the code with minimum wasted time. Instructions for importing an environment are found below.


## Quickstart


In PyCharm at the bottom of the screen click on the terminal (not IPython Console), or your command line if you're not going to use PyCharm.


Verify install and update with

```Bash
conda info
conda update -n base conda
```

To create an environment (NB: we may change to Python3.6 at some point if we ever integrate any MATLAB scripts)

```Bash
conda create --name ENVNAME python=3.7
````

To activate the environment (actually use it in the terminal)

```Bash
conda activate ENVNAME
```

To list the environments on this machine:

```Bash
conda info --envs
```

[Cheatsheet](https://docs.anaconda.com/anaconda/user-guide/cheatsheet/) tells you literally everything you need to know. (tip: save it to a pdf on your desktop)


## Packages


Then you can specify, seperately from the rest of your pc, the packages you want to be installed/used when running scripts when this environment is activated. To search for packages go to anaconda.org and just type in the name you want. NB: Don't use pip inside conda


```Bash
conda install numpy
```

If that doesn't work conda-forge usually will


```Bash
conda install -c conda-forge numpy
```

[Cheatsheet](https://docs.anaconda.com/anaconda/user-guide/cheatsheet/) tells you literally everything you need to know. (tip: save it to a pdf on your desktop)


## Sharing/Importing Environment


NB: recommended environment name currently is pw_env

Copy an environment (only to your pc):

```Bash
conda create --clone ENVNAME --name NEWENVNAME
```

Export a YAML file that can be used on Windows, Linux, macOS

```Bash
conda env export --name ENVNAME > envname.yml
```

Create an environment from a YAML file

```Bash
conda env create --file envname.yml
```

In PyCharm, you use environments slightly differently. In the PyCharm terminal you can use conda normally (activate the environment each time you open terminal). Otherwise, you need to point PyCharm to the environment. To do this you still need to create the environment from a yaml file as above or otherwise, and then in PyCharm you can find & select it for the current project.


# Basics


Code is read more than written. I think well written code will have as much commenting as code. We are not Bing, we aren't going for any speed awards that will care about a couple of extra lines in a text file. [Just Do it](https://bit.ly/31I7rQy)

You might like tabs, or 2 spaces, or something wacky. For Python, set your text editor to convert tabs to 4 spaces.
Seperating code with block comments for easy readability + useful for seperating distinct concepts (use # ==== convention)

We've decided to choose the convention of 100 character linewidths. This is not the PyCharm default (120)

Module-wide design concepts - what file is actually ran? Where is it run from? Command line vs IDE vs simple GUI, what needs to be visible to the user for them to operate? - Minimise and simplify this

Object structure should not be cyclic. Parent classes should not be referenced in children, and data should be passed back to parents via methods.

Recommended:
- Install Black, an autoformatter for your code to make git/diffs easier.
[link](https://github.com/psf/black) including instructions for PyCharm setup. At the end of the arguments ($FilePath$) you can further specify -l 100 to set the linewidth to be 100 characters


## Naming


- [PEP8](https://www.python.org/dev/peps/pep-0008/)+
- Lower case for variable names and functions, seperating words with underscores
- Classes and Types are **C**amel**C**ase
- Global variables (which we won't have cause we're really just very organised) should be __global__
- Constants should be defined on a Modular level, and should be all capitalized with underscores seperating words.
- Especially where the purpose of a variable is not clear, appending a descriptor of it's type (etc) is very useful. For example parameters_list as opposed to params might help the reader a lot.
    - use the convention of 'list' not 'lst' (lst looks like 1st, and most of the code is in the form of 'list' already)


## Documentation


- module docstrings - [dewit](https://bit.ly/2Mloft1)
- function docstrings - [dewit](https://bit.ly/2H8GnCr)

- ok at least have docstrings in each worker or 'parent' classes that describe the intention. For example fit worker needs a docstring, but lorentzian might not.

- Perhaps in a git wiki at some point, for now just in these markdown pages

- example py files, seperate from main file (i.e. designed so that they are always working)


## Global Design


- The global object structure will be placed here at a later point. In general there will be a WidefieldProcessor object that takes 'workers' to do specific jobs, that are written such that they can be used generally in other situations
- PlotWorker
- FitWorker
- DataWorker (e.g reading & transforming data)



