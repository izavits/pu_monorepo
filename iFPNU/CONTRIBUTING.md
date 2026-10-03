# Contributing Guide

Thanks for your interest in contributing to this project! This document aims to
serve as a friendly guide for making your first contribution.

## Developing

You will need:

- Python 3
- A virtual environment
- The dependencies described in requirements.txt
- The dataset

### Cloning the project

```sh
git clone git@github.com:izavits/pu_monorepo.git
cd pu_monorepo/iFPNU
```

### Installing

- Create your virtual environment and install the required dependencies:

```
virtualenv -p `which python3` venv
source venv/bin/activate
pip install -r requirements.txt
``` 


## Creating a pull request

When sending a pull request, consider the following guidelines:

- Write a concise commit message explaining your changes.

- If applies, write more descriptive information in the commit body.

- Refer to the issue/s your pull request fixes (if there are issues in the gitlab repo).

- Write a descriptive pull request title.

- Squash commits when possible.

Before your pull request can be merged, the following conditions must hold:

- All the tests pass. Make sure you run pytest after commiting your work and before sending the pull request.

- The coding style aligns with the project's convention. PEP-8 is used as a guide with minor modifications.

- Your changes are confirmed to be working.


**Working on your first Pull Request?** You can learn how from this *free* series [How to Contribute to an Open Source Project on GitHub](https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github)
