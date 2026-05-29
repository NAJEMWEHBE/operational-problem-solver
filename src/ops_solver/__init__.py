"""operational-problem-solver: multi-team consensus + ELO agentic engine.

Three worker teams orchestrated to solve a problem:
  - Intelligence (Crow): parallel context gathering -> unified ContextManifest
  - Execution (Finch): parallel isolated solution workers -> WorkerArtifact[]
  - Quality Assurance: a consensus stability gate, then an ELO tournament that
    ranks the surviving solutions head-to-head and ships the winner.
"""

__version__ = "0.1.0"
