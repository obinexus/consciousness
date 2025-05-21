"""
Bayesian DAG-Based Cognitive Simulation System

A framework for modeling cognitive processes as a Directed Acyclic Graph (DAG) 
with Bayesian inference, supporting both FilterFlash and FlashFilter thinking.

Compatible with https://github.com/obinexus/consciousness
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum, auto
import logging
from typing import Dict, List, Tuple, Callable, Optional, Union, Set
import math

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('cognitive_system')

class NodeType(Enum):
    """Types of cognitive nodes in the system"""
    INPUT = auto()         # Raw sensory or data input
    FEATURE = auto()       # Extracted feature from input
    CATEGORY = auto()      # Classification or categorization
    JUDGMENT = auto()      # Decision or judgment
    FEEDBACK = auto()      # Evaluation of output
    ABSTRACTION = auto()   # Higher-level abstractions
    HYPOTHESIS = auto()    # Top-down hypothesis

class InstructionType(Enum):
    """Instruction types that can be encoded in nodes"""
    CLASSIFY = auto()      # Classify input into categories
    DECOMPOSE = auto()     # Break down complex concepts
    WEIGH = auto()         # Evaluate importance/relevance
    ALIGN = auto()         # Align with existing knowledge
    ABSTRACT = auto()      # Generate higher-level concepts
    VERIFY = auto()        # Verify a hypothesis
    ADJUST = auto()        # Adjust weights or probabilities

class ProcessingMode(Enum):
    """Direction of cognitive processing"""
    FILTER_TO_FLASH = auto()  # Bottom-up processing (sensory  meaning)
    FLASH_TO_FILTER = auto()  # Top-down processing (insight  verification)

class CognitiveNode:
    """
    Represents a node in the cognitive DAG, which can be a question,
    feature, classification, or judgment.
    """
    
    def __init__(self, 
                 node_id: str, 
                 node_type: NodeType,
                 instruction: InstructionType,
                 prior_distribution: Dict = None,
                 update_function: Callable = None,
                 threshold: float = 0.85):
        """
        Initialize a cognitive node.
        
        Args:
            node_id: Unique identifier for this node
            node_type: Type of cognitive node (input, feature, etc.)
            instruction: Type of instruction encoded in this node
            prior_distribution: Initial probability distribution
            update_function: Custom function for Bayesian updates
            threshold: Confidence threshold for gamma flash
        """
        self.node_id = node_id
        self.node_type = node_type
        self.instruction = instruction
        
        # Initialize prior distribution (default uniform if none provided)
        self.prior_distribution = prior_distribution or {'default': 1.0}
        self.posterior_distribution = self.prior_distribution.copy()
        
        # Tracking confidence and evidence
        self.confidence = 0.0
        self.evidence_history = []
        
        # Custom update function or use default
        self.update_function = update_function
        
        # Gamma flash threshold
        self.threshold = threshold
        
        # Node state tracking
        self.activation_count = 0
        self.last_activated = None
        self.has_flashed = False
    
    def update_belief(self, evidence: Dict[str, float]) -> float:
        """
        Update posterior belief based on new evidence using Bayesian inference.
        
        Args:
            evidence: Dictionary mapping outcomes to likelihoods
            
        Returns:
            New confidence level
        """
        if self.update_function is not None:
            # Use custom update function if provided
            self.posterior_distribution = self.update_function(
                self.posterior_distribution, evidence
            )
        else:
            # Default naive Bayes update
            # P(A|B) ? P(B|A) * P(A)
            posterior = {}
            normalization = 0
            
            for outcome, prior in self.prior_distribution.items():
                if outcome in evidence:
                    likelihood = evidence[outcome]
                    posterior[outcome] = likelihood * prior
                    normalization += posterior[outcome]
                else:
                    posterior[outcome] = 0
            
            # Normalize
            if normalization > 0:
                for outcome in posterior:
                    posterior[outcome] /= normalization
            
            self.posterior_distribution = posterior
        
        # Calculate confidence (highest probability or entropy-based)
        if self.posterior_distribution:
            self.confidence = max(self.posterior_distribution.values())
        else:
            self.confidence = 0.0
        
        # Record evidence
        self.evidence_history.append(evidence)
        self.activation_count += 1
        self.last_activated = 0  # could be timestamp
        
        return self.confidence
    
    def check_flash(self) -> bool:
        """
        Check if node's confidence exceeds threshold, triggering a gamma flash.
        
        Returns:
            True if flash occurs, False otherwise
        """
        if self.confidence >= self.threshold and not self.has_flashed:
            self.has_flashed = True
            logger.info(f"GAMMA FLASH at node {self.node_id} with confidence {self.confidence:.4f}")
            return True
        return False
    
    def reset_flash(self):
        """Reset flash state to allow future flashes"""
        self.has_flashed = False
    
    def __repr__(self):
        return f"CognitiveNode({self.node_id}, {self.node_type}, conf={self.confidence:.2f})"


class BayesianInference:
    """
    Handles Bayesian inference and probability updates across the network.
    """
    
    @staticmethod
    def naive_bayes_update(prior: Dict[str, float], 
                          likelihood: Dict[str, float]) -> Dict[str, float]:
        """
        Update probabilities using Naive Bayes.
        
        Args:
            prior: Prior probability distribution
            likelihood: Likelihood based on evidence
            
        Returns:
            Updated probability distribution
        """
        posterior = {}
        normalization = 0
        
        for outcome, prior_prob in prior.items():
            if outcome in likelihood:
                posterior[outcome] = likelihood[outcome] * prior_prob
                normalization += posterior[outcome]
            else:
                posterior[outcome] = 0
        
        # Normalize
        if normalization > 0:
            for outcome in posterior:
                posterior[outcome] /= normalization
        
        return posterior
    
    @staticmethod
    def gaussian_update(prior_mean: float, 
                       prior_var: float,
                       evidence_mean: float,
                       evidence_var: float) -> Tuple[float, float]:
        """
        Update a Gaussian distribution with new evidence.
        
        Args:
            prior_mean: Mean of prior Gaussian
            prior_var: Variance of prior Gaussian
            evidence_mean: Mean of evidence Gaussian
            evidence_var: Variance of evidence Gaussian
            
        Returns:
            (posterior_mean, posterior_var)
        """
        # Precision is 1/variance
        prior_precision = 1.0 / prior_var
        evidence_precision = 1.0 / evidence_var
        
        # Posterior precision is sum of precisions
        posterior_precision = prior_precision + evidence_precision
        posterior_var = 1.0 / posterior_precision
        
        # Posterior mean is weighted by precisions
        posterior_mean = (prior_mean * prior_precision + 
                         evidence_mean * evidence_precision) / posterior_precision
        
        return posterior_mean, posterior_var
    
    @staticmethod
    def entropy(distribution: Dict[str, float]) -> float:
        """
        Calculate entropy of a probability distribution as a measure of uncertainty.
        
        Args:
            distribution: Probability distribution
            
        Returns:
            Entropy value (higher = more uncertain)
        """
        entropy = 0
        for p in distribution.values():
            if p > 0:  # Avoid log(0)
                entropy -= p * math.log2(p)
        return entropy


class GammaWaveDetector:
    """
    Detects and manages gamma-like bursts (flashes) in the cognitive network.
    """
    
    def __init__(self, graph, global_threshold: float = 0.85):
        """
        Initialize the gamma wave detector.
        
        Args:
            graph: NetworkX graph of cognitive nodes
            global_threshold: Default threshold for all nodes
        """
        self.graph = graph
        self.global_threshold = global_threshold
        self.flash_history = []
    
    def check_all_nodes(self) -> List[str]:
        """
        Check all nodes for potential gamma flashes.
        
        Returns:
            List of node IDs that flashed
        """
        flashed_nodes = []
        
        for node_id in self.graph.nodes:
            node = self.graph.nodes[node_id]['data']
            if node.check_flash():
                flashed_nodes.append(node_id)
                self.flash_history.append((node_id, node.confidence))
                
                # Trigger reweighting of adjacent nodes
                self._reweight_adjacent_nodes(node_id)
        
        return flashed_nodes
    
    def _reweight_adjacent_nodes(self, flashed_node_id: str):
        """
        Adjust weights of edges to/from the flashed node.
        
        Args:
            flashed_node_id: ID of node that experienced a flash
        """
        # Boost incoming edges
        for pred in self.graph.predecessors(flashed_node_id):
            edge_data = self.graph[pred][flashed_node_id]
            # Increase weight by 10%
            edge_data['weight'] = min(1.0, edge_data['weight'] * 1.1)
        
        # Boost outgoing edges
        for succ in self.graph.successors(flashed_node_id):
            edge_data = self.graph[flashed_node_id][succ]
            # Increase weight by 10%
            edge_data['weight'] = min(1.0, edge_data['weight'] * 1.1)
    
    def get_flash_summary(self) -> Dict:
        """
        Generate summary of gamma flash activity.
        
        Returns:
            Dictionary with flash statistics
        """
        if not self.flash_history:
            return {"total_flashes": 0}
        
        # Get counts by node
        node_counts = {}
        for node_id, _ in self.flash_history:
            node_counts[node_id] = node_counts.get(node_id, 0) + 1
        
        # Find most active nodes
        sorted_nodes = sorted(node_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "total_flashes": len(self.flash_history),
            "unique_nodes_flashed": len(node_counts),
            "most_active_nodes": sorted_nodes[:3] if sorted_nodes else [],
            "last_flash": self.flash_history[-1] if self.flash_history else None
        }


class CognitiveDAG:
    """
    Main class implementing the Bayesian DAG cognitive model.
    """
    
    def __init__(self):
        """Initialize the cognitive DAG model."""
        # Initialize graph
        self.graph = nx.DiGraph()
        
        # Initialize components
        self.inference_engine = BayesianInference()
        self.gamma_detector = None  # Will be created after graph is built
        
        # Tracking
        self.processing_mode = ProcessingMode.FILTER_TO_FLASH
        self.execution_history = []
        self.current_path = []
    
    def add_node(self, node: CognitiveNode) -> str:
        """
        Add a cognitive node to the graph.
        
        Args:
            node: CognitiveNode instance
            
        Returns:
            Node ID
        """
        self.graph.add_node(node.node_id, data=node)
        return node.node_id
    
    def add_edge(self, from_id: str, to_id: str, weight: float = 0.5) -> None:
        """
        Add a weighted edge between nodes.
        
        Args:
            from_id: Source node ID
            to_id: Target node ID
            weight: Edge weight (transition probability)
        """
        self.graph.add_edge(from_id, to_id, weight=weight)
    
    def initialize(self):
        """Initialize components after graph is built."""
        self.gamma_detector = GammaWaveDetector(self.graph)
    
    def filter_to_flash(self, input_data: Dict[str, float], 
                       input_node_ids: List[str]) -> List[str]:
        """
        Execute bottom-up processing (FilterFlash).
        
        Args:
            input_data: Input evidence/data
            input_node_ids: List of input node IDs
            
        Returns:
            List of nodes that flashed
        """
        self.processing_mode = ProcessingMode.FILTER_TO_FLASH
        self.current_path = []
        
        # Process input nodes
        for node_id in input_node_ids:
            if node_id in self.graph.nodes:
                node = self.graph.nodes[node_id]['data']
                node.update_belief(input_data)
                self.current_path.append(node_id)
        
        # Process subsequent layers (breadth-first)
        visited = set(input_node_ids)
        frontier = list(input_node_ids)
        
        while frontier:
            current_id = frontier.pop(0)
            
            # Process successors
            for succ_id in self.graph.successors(current_id):
                if succ_id not in visited:
                    # Collect evidence from predecessors
                    evidence = {}
                    for pred_id in self.graph.predecessors(succ_id):
                        pred_node = self.graph.nodes[pred_id]['data']
                        # Use max probability as evidence strength
                        if pred_node.posterior_distribution:
                            max_outcome = max(pred_node.posterior_distribution.items(), 
                                             key=lambda x: x[1])
                            evidence[max_outcome[0]] = max_outcome[1]
                    
                    # Update node with evidence
                    succ_node = self.graph.nodes[succ_id]['data']
                    succ_node.update_belief(evidence)
                    
                    # Add to visited and frontier
                    visited.add(succ_id)
                    frontier.append(succ_id)
                    self.current_path.append(succ_id)
        
        # Check for gamma flashes
        flashed_nodes = self.gamma_detector.check_all_nodes()
        
        # Record execution
        self.execution_history.append({
            'mode': self.processing_mode,
            'path': self.current_path.copy(),
            'flashes': flashed_nodes.copy()
        })
        
        return flashed_nodes
    
    def flash_to_filter(self, hypothesis_node_id: str, 
                       hypothesis_data: Dict[str, float]) -> List[str]:
        """
        Execute top-down processing (FlashFilter).
        
        Args:
            hypothesis_node_id: Starting hypothesis node
            hypothesis_data: Initial hypothesis beliefs
            
        Returns:
            List of nodes that flashed during verification
        """
        self.processing_mode = ProcessingMode.FLASH_TO_FILTER
        self.current_path = []
        
        # Set initial hypothesis
        if hypothesis_node_id in self.graph.nodes:
            node = self.graph.nodes[hypothesis_node_id]['data']
            node.update_belief(hypothesis_data)
            self.current_path.append(hypothesis_node_id)
        else:
            return []
        
        # Process layers downward (breadth-first)
        visited = {hypothesis_node_id}
        frontier = [hypothesis_node_id]
        
        while frontier:
            current_id = frontier.pop(0)
            current_node = self.graph.nodes[current_id]['data']
            
            # Get predecessors (moving down from hypothesis to supporting evidence)
            for pred_id in self.graph.predecessors(current_id):
                if pred_id not in visited:
                    # Generate expectations based on hypothesis
                    expectation = {}
                    # Convert posterior to expectations for predecessors
                    for outcome, prob in current_node.posterior_distribution.items():
                        expectation[outcome] = prob
                    
                    # Update node with expectations
                    pred_node = self.graph.nodes[pred_id]['data']
                    pred_node.update_belief(expectation)
                    
                    # Add to visited and frontier
                    visited.add(pred_id)
                    frontier.append(pred_id)
                    self.current_path.append(pred_id)
        
        # Check for gamma flashes
        flashed_nodes = self.gamma_detector.check_all_nodes()
        
        # Record execution
        self.execution_history.append({
            'mode': self.processing_mode,
            'path': self.current_path.copy(),
            'flashes': flashed_nodes.copy()
        })
        
        return flashed_nodes
    
    def update_from_feedback(self, feedback_data: Dict[str, float], 
                           feedback_node_ids: List[str]) -> None:
        """
        Update network weights and priors based on feedback.
        
        Args:
            feedback_data: Feedback values
            feedback_node_ids: Nodes to receive feedback
        """
        # Apply feedback to specified nodes
        for node_id in feedback_node_ids:
            if node_id in self.graph.nodes:
                node = self.graph.nodes[node_id]['data']
                node.update_belief(feedback_data)
                
                # Reset flash state to allow new flashes
                node.reset_flash()
        
        # Update edge weights based on confidence alignment
        for edge in self.graph.edges:
            source_id, target_id = edge
            source_node = self.graph.nodes[source_id]['data']
            target_node = self.graph.nodes[target_id]['data']
            
            # If both nodes activated recently
            if (source_node.activation_count > 0 and 
                target_node.activation_count > 0):
                
                # Check if confidences align
                confidence_diff = abs(source_node.confidence - target_node.confidence)
                
                # If confidences are similar, strengthen connection
                if confidence_diff < 0.2:
                    self.graph[source_id][target_id]['weight'] = min(
                        1.0, self.graph[source_id][target_id]['weight'] * 1.05
                    )
                # If very different, weaken connection
                elif confidence_diff > 0.5:
                    self.graph[source_id][target_id]['weight'] = max(
                        0.05, self.graph[source_id][target_id]['weight'] * 0.95
                    )
    
    def visualize(self, highlight_path: bool = True, 
                highlight_flashes: bool = True, 
                filename: str = None) -> None:
        """
        Visualize the cognitive network.
        
        Args:
            highlight_path: Whether to highlight the current execution path
            highlight_flashes: Whether to highlight nodes that flashed
            filename: If provided, save visualization to file
        """
        plt.figure(figsize=(12, 8))
        
        # Create position layout
        pos = nx.spring_layout(self.graph)
        
        # Draw basic graph
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3, 
                             width=[self.graph[u][v]['weight'] * 3 for u, v in self.graph.edges])
        
        # Prepare node colors based on type
        node_colors = []
        for node_id in self.graph.nodes:
            node = self.graph.nodes[node_id]['data']
            if node.node_type == NodeType.INPUT:
                node_colors.append('skyblue')
            elif node.node_type == NodeType.FEATURE:
                node_colors.append('lightgreen')
            elif node.node_type == NodeType.CATEGORY:
                node_colors.append('orange')
            elif node.node_type == NodeType.JUDGMENT:
                node_colors.append('salmon')
            else:
                node_colors.append('lightgray')
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, alpha=0.8, 
                             node_size=[300 + self.graph.nodes[n]['data'].confidence * 200 
                                       for n in self.graph.nodes])
        
        # Highlight current path if requested
        if highlight_path and self.current_path:
            path_edges = [(self.current_path[i], self.current_path[i+1]) 
                         for i in range(len(self.current_path)-1) 
                         if (self.current_path[i], self.current_path[i+1]) in self.graph.edges]
            nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, 
                                 edge_color='blue', width=2.5)
        
        # Highlight flashes if requested
        if highlight_flashes:
            flashed_nodes = [n for n in self.graph.nodes 
                           if self.graph.nodes[n]['data'].has_flashed]
            if flashed_nodes:
                nx.draw_networkx_nodes(self.graph, pos, nodelist=flashed_nodes, 
                                     node_color='yellow', node_size=500, alpha=1.0)
        
        # Add node labels
        node_labels = {node_id: f"{node_id}\n{self.graph.nodes[node_id]['data'].confidence:.2f}" 
                      for node_id in self.graph.nodes}
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=8)
        
        # Set title based on current mode
        if self.processing_mode == ProcessingMode.FILTER_TO_FLASH:
            plt.title("Cognitive DAG - FilterFlash Mode")
        else:
            plt.title("Cognitive DAG - FlashFilter Mode")
            
        plt.axis('off')
        
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
    
    def get_execution_summary(self) -> Dict:
        """
        Generate summary of execution history.
        
        Returns:
            Dictionary with execution statistics
        """
        if not self.execution_history:
            return {"total_executions": 0}
        
        # Count modes
        filter_to_flash_count = sum(1 for exec in self.execution_history 
                                 if exec['mode'] == ProcessingMode.FILTER_TO_FLASH)
        flash_to_filter_count = len(self.execution_history) - filter_to_flash_count
        
        # Count flashes
        total_flashes = sum(len(exec['flashes']) for exec in self.execution_history)
        
        # Find most common path nodes
        path_nodes = {}
        for exec in self.execution_history:
            for node in exec['path']:
                path_nodes[node] = path_nodes.get(node, 0) + 1
        
        sorted_path_nodes = sorted(path_nodes.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "total_executions": len(self.execution_history),
            "filter_to_flash_count": filter_to_flash_count,
            "flash_to_filter_count": flash_to_filter_count,
            "total_flashes": total_flashes,
            "most_common_path_nodes": sorted_path_nodes[:5] if sorted_path_nodes else [],
            "most_recent_mode": self.execution_history[-1]['mode'] if self.execution_history else None
        }


# Example implementation follows - color classification task

def build_color_classification_model() -> CognitiveDAG:
    """
    Build a network for color classification from RGB inputs.
    
    Returns:
        Configured cognitive DAG model
    """
    # Create cognitive DAG
    model = CognitiveDAG()
    
    # Create nodes
    # Input nodes for RGB channels
    r_input = CognitiveNode("r_input", NodeType.INPUT, InstructionType.CLASSIFY,
                           prior_distribution={"low": 0.5, "high": 0.5})
    g_input = CognitiveNode("g_input", NodeType.INPUT, InstructionType.CLASSIFY,
                           prior_distribution={"low": 0.5, "high": 0.5})
    b_input = CognitiveNode("b_input", NodeType.INPUT, InstructionType.CLASSIFY,
                           prior_distribution={"low": 0.5, "high": 0.5})
    
    # Feature extraction nodes
    intensity = CognitiveNode("intensity", NodeType.FEATURE, InstructionType.ABSTRACT,
                             prior_distribution={"dark": 0.33, "medium": 0.34, "bright": 0.33})
    warmth = CognitiveNode("warmth", NodeType.FEATURE, InstructionType.ABSTRACT,
                          prior_distribution={"cool": 0.5, "warm": 0.5})
    
    # Color category nodes
    red = CognitiveNode("red", NodeType.CATEGORY, InstructionType.CLASSIFY,
                       prior_distribution={"yes": 0.2, "no": 0.8})
    green = CognitiveNode("green", NodeType.CATEGORY, InstructionType.CLASSIFY,
                         prior_distribution={"yes": 0.2, "no": 0.8})
    blue = CognitiveNode("blue", NodeType.CATEGORY, InstructionType.CLASSIFY,
                        prior_distribution={"yes": 0.2, "no": 0.8})
    yellow = CognitiveNode("yellow", NodeType.CATEGORY, InstructionType.CLASSIFY,
                          prior_distribution={"yes": 0.2, "no": 0.8})
    
    # Higher-level judgment nodes
    primary = CognitiveNode("primary_color", NodeType.JUDGMENT, InstructionType.CLASSIFY,
                           prior_distribution={"yes": 0.5, "no": 0.5})
    warm_color = CognitiveNode("warm_color", NodeType.JUDGMENT, InstructionType.CLASSIFY,
                              prior_distribution={"yes": 0.5, "no": 0.5})
    
    # Abstract hypothesis node (for top-down processing)
    color_hypothesis = CognitiveNode("color_hypothesis", NodeType.HYPOTHESIS, InstructionType.DECOMPOSE,
                                    prior_distribution={"red": 0.25, "green": 0.25, "blue": 0.25, "yellow": 0.25})
    
    # Feedback node
    feedback = CognitiveNode("feedback", NodeType.FEEDBACK, InstructionType.ADJUST,
                            prior_distribution={"correct": 0.5, "incorrect": 0.5})
    
    # Add nodes to graph
    model.add_node(r_input)
    model.add_node(g_input)
    model.add_node(b_input)
    model.add_node(intensity)
    model.add_node(warmth)
    model.add_node(red)
    model.add_node(green)
    model.add_node(blue)
    model.add_node(yellow)
    model.add_node(primary)
    model.add_node(warm_color)
    model.add_node(color_hypothesis)
    model.add_node(feedback)
    
    # Add edges with initial weights
    # Input to features
    model.add_edge("r_input", "intensity", 0.7)
    model.add_edge("g_input", "intensity", 0.5)
    model.add_edge("b_input", "intensity", 0.5)
    model.add_edge("r_input", "warmth", 0.8)
    model.add_edge("g_input", "warmth", 0.4)
    model.add_edge("b_input", "warmth", 0.2)
    
    # Features to categories
    model.add_edge("r_input", "red", 0.9)
    model.add_edge("g_input", "green", 0.9)
    model.add_edge("b_input", "blue", 0.9)
    model.add_edge("r_input", "yellow", 0.5)
    model.add_edge("g_input", "yellow", 0.5)
    model.add_edge("warmth", "red", 0.7)
    model.add_edge("warmth", "yellow", 0.7)
    
    # Categories to judgments
    model.add_edge("red", "primary_color", 0.8)
    model.add_edge("green", "primary_color", 0.8)
    model.add_edge("blue", "primary_color", 0.8)
    model.add_edge("yellow", "primary_color", 0.4)
    model.add_edge("red", "warm_color", 0.9)
    model.add_edge("yellow", "warm_color", 0.9)
    
    # Hypothesis to categories (top-down)
    model.add_edge("color_hypothesis", "red", 0.7)
    model.add_edge("color_hypothesis", "green", 0.7)
    model.add_edge("color_hypothesis", "blue", 0.7)
    model.add_edge("color_hypothesis", "yellow", 0.7)
    
    # Feedback connections
    model.add_edge("primary_color", "feedback", 0.5)
    model.add_edge("warm_color", "feedback", 0.5)
    model.add_edge("red", "feedback", 0.5)
    model.add_edge("green", "feedback", 0.5)
    model.add_edge("blue", "feedback", 0.5)
    model.add_edge("yellow", "feedback", 0.5)
    
    # Initialize model
    model.initialize()
    
    return model


def example_color_classification():
    """Run an example of the color classification system."""
    # Build the model
    model = build_color_classification_model()
    
    # Define some test inputs
    # Reddish color
    red_input = {
        "high": 0.9,  # High red
        "low": 0.1
    }
    
    green_input = {
        "high": 0.2,  # Low green
        "low": 0.8
    }
    
    blue_input = {
        "high": 0.1,  # Low blue
        "low": 0.9
    }
    
    # Example 1: FilterFlash processing (bottom-up)
    print("\n=== FilterFlash Classification (Bottom-Up) ===")
    
    # Process inputs
    input_nodes = ["r_input", "g_input", "b_input"]
    flashed_nodes = model.filter_to_flash(
        {"high": 0.9, "low": 0.1}, ["r_input"]
    )
    flashed_nodes += model.filter_to_flash(
        {"high": 0.2, "low": 0.8}, ["g_input"]
    )
    flashed_nodes += model.filter_to_flash(
        {"high": 0.1, "low": 0.9}, ["b_input"]
    )
    
    print(f"Flashed nodes: {flashed_nodes}")
    
    # Visualize the results
    model.visualize(filename="filter_to_flash.png")
    
    # Print confidences
    print("\nNode Confidences:")
    for node_id in model.graph.nodes:
        node = model.graph.nodes[node_id]['data']
        print(f"  {node_id}: {node.confidence:.4f} - {node.posterior_distribution}")
    
    # Example 2: FlashFilter processing (top-down)
    print("\n=== FlashFilter Verification (Top-Down) ===")
    
    # Start with a hypothesis about the color
    hypothesis = {
        "red": 0.7,
        "green": 0.1,
        "blue": 0.1,
        "yellow": 0.1
    }
    
    # Execute top-down processing
    flashed_nodes = model.flash_to_filter("color_hypothesis", hypothesis)
    print(f"Flashed nodes: {flashed_nodes}")
    
    # Visualize the results
    model.visualize(filename="flash_to_filter.png")
    
    # Example 3: Learning from feedback
    print("\n=== Learning from Feedback ===")
    
    # Provide positive feedback
    feedback_data = {
        "correct": 0.9,
        "incorrect": 0.1
    }
    
    model.update_from_feedback(feedback_data, ["feedback"])
    
    # Print updated confidences and weights
    print("\nUpdated Edge Weights:")
    for edge in model.graph.edges:
        source, target = edge
        weight = model.graph[source][target]['weight']
        print(f"  {source}  {target}: {weight:.4f}")
    
    # Get summaries
    flash_summary = model.gamma_detector.get_flash_summary()
    execution_summary = model.get_execution_summary()
    
    print("\nGamma Flash Summary:")
    for key, value in flash_summary.items():
        print(f"  {key}: {value}")
    
    print("\nExecution Summary:")
    for key, value in execution_summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # Run the example
    example_color_classification()
