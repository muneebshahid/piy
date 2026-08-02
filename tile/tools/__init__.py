"""Built-in tools for the default agent."""

from tile.tool_truncation import ToolOutputDetails
from tile.tools.bash import BashDetails
from tile.tools.bash import tool as bash_tool
from tile.tools.complete import CompleteDetails
from tile.tools.complete import tool as complete_tool
from tile.tools.edit import EditDetails
from tile.tools.edit import tool as edit_tool
from tile.tools.fail import FailDetails
from tile.tools.fail import tool as fail_tool
from tile.tools.find import FindDetails
from tile.tools.find import tool as find_tool
from tile.tools.grep import GrepDetails
from tile.tools.grep import tool as grep_tool
from tile.tools.ls import LsDetails
from tile.tools.ls import tool as ls_tool
from tile.tools.read import ReadDetails
from tile.tools.read import tool as read_tool
from tile.tools.write import tool as write_tool
from tile.types.tools import ToolDefinition

__all__ = [
    "BUILTIN_TOOLS",
    "BashDetails",
    "CompleteDetails",
    "EditDetails",
    "FailDetails",
    "FindDetails",
    "GrepDetails",
    "LsDetails",
    "ReadDetails",
    "ToolOutputDetails",
    "complete_tool",
    "fail_tool",
]

BUILTIN_TOOLS: tuple[ToolDefinition, ...] = (
    read_tool,
    bash_tool,
    edit_tool,
    grep_tool,
    find_tool,
    ls_tool,
    write_tool,
)
