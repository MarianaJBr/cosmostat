from dataclasses import dataclass

from click import ClickException
from rich.console import Console

from .util import console


@dataclass
class CLIError(ClickException):
    """Represent a CLI Exception."""
    message: str
    console: Console = console
    label: str = "[red]Error[/]"

    def __post_init__(self):
        """"""
        super().__init__(self.message)

    def show(self, file=None):
        """"""
        message = self.format_message()
        self.console.print(f"{self.label}: {message}")
