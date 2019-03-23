# Learning
Project for the Deep Learning class

## Getting started

### Cloning

[Install `git lfs`](https://git-lfs.github.com) (on macOS: `brew install git-lfs`), **and then** run:

```bash
git lfs install
```

If you installed `git-lfs` after cloning the repo, you can use the following command to download LFS files:

```bash
git lfs fetch
```

### Setting up an environment *(Optional)*

If you set up a virtual environment and store it in the root folder, make sure 
not to add it to git to name it like one of those options in the `.gitignore`:

```
env/
venv/
ENV/
env.bak/
venv.bak/
```

### Installing packages

```bash
pip install -r requirements.txt
```
