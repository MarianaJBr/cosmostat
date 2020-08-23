import pathlib
import sys
import typing as t

from dask.callbacks import Callback
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

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


# ProgressBar columns.
columns = (
    TextColumn("[progress.description]{task.description}"),
    BarColumn(bar_width=None),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeRemainingColumn(),
)
progress = Progress(*columns, console=console)


class ProgressBar(Callback):
    """Progressbar for dask computations."""

    def _start_state(self, dsk, state):
        """"""
        total = sum(len(state[k]) for k in ["ready", "waiting", "running"])
        self._rich_progress = progress
        self._progress_task = progress.add_task("[red]Progress", total=total)
        progress.start()

    def _posttask(self, key, result, dsk, state, worker_id):
        """"""
        progress_task = self._progress_task
        self._rich_progress.update(progress_task, advance=1)

    def _finish(self, dsk, state, errored):
        self._rich_progress.stop()
