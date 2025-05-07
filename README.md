## Install uv

Uv can be installed as follows, depending on your operating system.

<br>

**macOS and Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

<br>

**Windows**

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | more"
```

## 2. Install Python packages and dependencies

To install all required packages from a `pyproject.toml` file (such as the one located at the top level of this GitHub repository), run the following command, assuming the file is in the same directory as your terminal session:

```bash
uv sync --dev --python 3.10.4
```

> **Note:**
> If you do not have Python 3.11 available on your system, uv will download and install it for you.
> I recommend using a Python version that is at least 1-3 versions older than the most recent release to ensure PyTorch compatibility. For example, if the most recent version is Python 3.13, I recommend using version 3.10, 3.11, 3.12. You can find out the most recent Python version by visiting [python.org](https://www.python.org/downloads/).

> **Note:**
> If you have problems with the following commands above due to certain dependencies (for example, if you are using Windows), you can always fall back to regular pip:
> `uv add pip`
> `uv run python -m pip install -U -r requirements.txt`
>
> Since the TensorFo




Note that the `uv sync` command above will create a separate virtual environment via the `.venv` subfolder. (In case you want to delete your virtual environment to start from scratch, you can simply delete the `.venv` folder.)

You can install new packages, that are not specified in the `pyproject.toml` via `uv add`, for example:

```bash
uv add packaging
```

And you can remove packages via `uv remove`, for example,

```bash
uv remove packaging
```

## 3. Run Python code

<br>

Your environment should now be ready to run the code in the repository.

```bash
uv run gpt-from-scratch
```

<br>

**Skipping the `uv run` command**

If you find typing `uv run` cumbersome, you can manually activate the virtual environment as described below.

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows (PowerShell):

```bash
.venv\Scripts\activate
```

Then, you can run scripts via

```bash
python script.py
```
