"""
Orchestration module for SOCIA-CAMEL system.
Contains workflow management and agent coordination components.
"""

from .workflow_manager import WorkflowManager
from .container import AgentContainer

__all__ = [
    'WorkflowManager',
    'AgentContainer'
]
