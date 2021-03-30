import importlib
import os
import pathlib
import sys
import typing as t

from dask.callbacks import Callback
from rich.console import Console, RenderableType
from rich.padding import Padding
from rich.progress import BarColumn, Progress, TextColumn

# Rich output console instances.
console = Console()
err_console = Console(file=sys.stderr)


def parse_file_and_group(path: t.Union[str, pathlib.Path]):
    """Locates the HDF5 file and group from a HDF5 file name.

    See https://support.hdfgroup.org/HDF5/Tutor/cmdtoolview.html#group.
    """
    path = pathlib.Path(path).resolve()
    file = path
    parent = path.parent
    group_parts = []
    while parent != file:
        if path.exists() and path.is_dir():
            raise IsADirectoryError
        else:
            group_parts.insert(0, file.name)
            if parent.exists() and not parent.is_dir():
                file = parent
                break
        file = parent
        parent = file.parent
    return file, "/".join(group_parts)


class RichProgressBar(Progress):
    """A slightly modified rich progress bar."""

    def get_renderables(self) -> t.Iterable[RenderableType]:
        """"""
        yield Padding(self.make_tasks_table(self.tasks), (1, 1))


# ProgressBar columns.
columns = (
    TextColumn("[progress.description]{task.description}"),
    BarColumn(bar_width=None),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TextColumn("|"),
    TextColumn("[progress.remaining]Elapsed time: {task.elapsed:>.2f}s"),
)


class DaskProgressBar(Callback):
    """Progress bar for dask computations."""

    def _start_state(self, dsk, state):
        """"""
        total = sum(len(state[k]) for k in ["ready", "waiting", "running"])
        progress = RichProgressBar(*columns, console=console,
                                   auto_refresh=False)
        self.rich_progress = progress
        self.main_task = progress.add_task("[red]Progress", total=total)
        progress.start()

    def _posttask(self, key, result, dsk, state, worker_id):
        """"""
        progress_task = self.main_task
        self.rich_progress.update(progress_task, advance=1)
        self.rich_progress.refresh()

    def _finish(self, dsk, state, errored):
        self.rich_progress.stop()


# Environment variables that control the location of the external models.
EXTRA_PATH_ENV = "COSMOSTAT_EXTRA_PATH"
ENTRY_POINT_ENV = "COSMOSTAT_EXTRA_ENTRY_POINT"
DEFAULT_ENTRY_POINT = "my_cosmostat_models"


def plug_external_models():
    """Load the external models inside a specified directory."""
    extra_path_env = os.getenv(EXTRA_PATH_ENV)
    if extra_path_env is None:
        extra_path = pathlib.Path.cwd()
    else:
        extra_path = pathlib.Path(extra_path_env).resolve()
    # Update PythonPath.
    sys.path.insert(0, str(extra_path))
    # We just import the module. The model and other options must be
    # registered in the module body.
    entry_point_env = os.getenv(ENTRY_POINT_ENV)
    entry_point = entry_point_env or DEFAULT_ENTRY_POINT
    try:
        return importlib.import_module(entry_point)
    except ImportError:
        return
