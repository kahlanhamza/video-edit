"""DirectorOS agent runtime — native tool execution engine (PLAN §3).

- `registry.ToolRegistry`   typed dispatch: Pydantic-validated args -> fn
- `editing_tools`           the director toolset over OTIO timelines
- `checkpoint.CheckpointStore`  durable timeline versions + tool audit log
- `agents.PlannerExecutor`  checkpoint-before-execute loop w/ auto-rollback
"""

from director_os.agent_runtime.agents import (
    AgentRole,
    AgentRun,
    LLMBackend,
    PlannerExecutor,
    ScriptedBackend,
    StepResult,
    ToolStep,
)
from director_os.agent_runtime.checkpoint import CheckpointError, CheckpointStore
from director_os.agent_runtime.editing_tools import (
    DirectorToolbox,
    ToolOperationError,
    build_director_registry,
)
from director_os.agent_runtime.registry import (
    ToolCall,
    ToolDispatchError,
    ToolRegistry,
    ToolSpec,
)
from director_os.agent_runtime.workshop_tools import register_workshop_tools

__all__ = [
    "AgentRole",
    "AgentRun",
    "CheckpointError",
    "CheckpointStore",
    "DirectorToolbox",
    "LLMBackend",
    "PlannerExecutor",
    "ScriptedBackend",
    "StepResult",
    "ToolCall",
    "ToolDispatchError",
    "ToolOperationError",
    "ToolRegistry",
    "ToolSpec",
    "ToolStep",
    "build_director_registry",
    "register_workshop_tools",
]
