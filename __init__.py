"""
Cognitive Filter-Flash DAG Framework

A Bayesian DAG-based cognitive simulation system that models
the Filter-Flash theory of consciousness.

This package provides:
- A Directed Acyclic Graph (DAG) structure for cognitive processing
- Support for both FilterFlash (bottom-up) and FlashFilter (top-down) modes
- Bayesian inference for probability-based reasoning
- Gamma wave detection for modeling "aha moments"
- Enhanced visualizations with entropy-based state encoding

Compatible with https://github.com/obinexus/consciousness
"""

# Import primary components for easy access
from .cognitive_system import (
    CognitiveDAG,
    CognitiveNode,
    NodeType,
    InstructionType,
    ProcessingMode,
    BayesianInference,
    GammaWaveDetector,
    build_color_classification_model
)

# Import visualization components
from .cognitive_visualizations import EnhancedCognitiveVisualizations

__version__ = "1.0.0"
__author__ = "Nnamdi Michael Okpala"
