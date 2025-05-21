"""
Enhanced Filter-Flash Network Visualizations and State Encoding

This module extends the Bayesian DAG Cognitive Simulation with advanced 
visualizations and state encoding methods that highlight both the objective
FilterFlash and subjective FlashFilter processing modes.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from io import BytesIO
import base64
from typing import Dict, List, Tuple, Optional, Set
import math

# Assume we have our cognitive_system module imported
# from cognitive_system import CognitiveDAG, NodeType, ProcessingMode


class EnhancedCognitiveVisualizations:
    """
    Enhanced visualization capabilities for the Filter-Flash cognitive model.
    """
    
    def __init__(self, cognitive_dag):
        """
        Initialize with a cognitive DAG instance.
        
        Args:
            cognitive_dag: CognitiveDAG instance to visualize
        """
        self.dag = cognitive_dag
        
        # Custom colormaps for different visualizations
        self.filter_flash_cmap = plt.cm.Blues
        self.flash_filter_cmap = plt.cm.Oranges
        self.entropy_cmap = LinearSegmentedColormap.from_list(
            'EntropyMap', ['darkgreen', 'yellow', 'red'], N=100)
    
    def create_filter_flash_representation(self, 
                                         filename: Optional[str] = None) -> nx.DiGraph:
        """
        Create and visualize a network representation focused on the FilterFlash dynamics.
        
        Args:
            filename: If provided, save visualization to file
            
        Returns:
            NetworkX DiGraph with FilterFlash properties
        """
        G = nx.DiGraph()
        
        # Create a layered representation based on node types
        layers = {
            NodeType.INPUT: 0,
            NodeType.FEATURE: 1, 
            NodeType.CATEGORY: 2,
            NodeType.JUDGMENT: 3,
            NodeType.ABSTRACTION: 4,
            NodeType.HYPOTHESIS: 5
        }
        
        # Add nodes with positions based on layer
        positions = {}
        node_confidence = {}
        node_entropy = {}
        node_flash_state = {}
        
        # Collect nodes by layer
        nodes_by_layer = {}
        for layer_type in layers.keys():
            nodes_by_layer[layer_type] = []
        
        # First pass: group nodes by layer
        for node_id in self.dag.graph.nodes:
            node = self.dag.graph.nodes[node_id]['data']
            layer = layers.get(node.node_type, 0)
            nodes_by_layer[node.node_type].append(node_id)
            
            # Add node to visualization graph with properties
            G.add_node(node_id, 
                      layer=layer, 
                      node_type=node.node_type,
                      instruction=node.instruction,
                      confidence=node.confidence,
                      has_flashed=node.has_flashed)
            
            # Calculate entropy
            entropy = self._calculate_entropy(node.posterior_distribution)
            node_entropy[node_id] = entropy
            node_confidence[node_id] = node.confidence
            node_flash_state[node_id] = node.has_flashed
        
        # Second pass: assign positions within each layer
        for layer_type, layer_nodes in nodes_by_layer.items():
            if not layer_nodes:
                continue
                
            layer = layers[layer_type]
            num_nodes = len(layer_nodes)
            
            for i, node_id in enumerate(layer_nodes):
                # Position nodes horizontally within their layer
                x_pos = (i + 1) / (num_nodes + 1)
                y_pos = 1.0 - (layer / (len(layers) + 1))
                positions[node_id] = (x_pos, y_pos)
        
        # Add edges
        for edge in self.dag.graph.edges:
            source, target = edge
            weight = self.dag.graph[source][target]['weight']
            G.add_edge(source, target, weight=weight)
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        
        # Draw edges with weights as widths and transparency
        edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
        nx.draw_networkx_edges(G, positions, width=edge_weights, alpha=0.6, 
                              edge_color='gray', arrows=True, arrowsize=15)
        
        # Draw nodes with size based on confidence and color based on layer
        node_sizes = [300 + node_confidence[n] * 300 for n in G.nodes()]
        node_colors = [layers[G.nodes[n]['node_type']] for n in G.nodes()]
        
        # Draw all nodes
        nx.draw_networkx_nodes(G, positions, node_size=node_sizes, 
                              node_color=node_colors, cmap=self.filter_flash_cmap,
                              alpha=0.8)
        
        # Highlight flashed nodes
        flashed_nodes = [n for n in G.nodes() if node_flash_state[n]]
        if flashed_nodes:
            nx.draw_networkx_nodes(G, positions, nodelist=flashed_nodes, 
                                  node_size=[node_sizes[list(G.nodes()).index(n)] + 100 
                                            for n in flashed_nodes],
                                  node_color='yellow', alpha=0.8)
        
        # Add labels with entropy and confidence
        node_labels = {node_id: f"{node_id}\nC:{node_confidence[node_id]:.2f}\nE:{node_entropy[node_id]:.2f}" 
                      for node_id in G.nodes()}
        nx.draw_networkx_labels(G, positions, labels=node_labels, font_size=8)
        
        # Add path highlight if available
        if self.dag.current_path:
            path_edges = [(self.dag.current_path[i], self.dag.current_path[i+1]) 
                         for i in range(len(self.dag.current_path)-1) 
                         if (self.dag.current_path[i], self.dag.current_path[i+1]) in G.edges()]
            if path_edges:
                nx.draw_networkx_edges(G, positions, edgelist=path_edges, 
                                      edge_color='blue', width=3.0, alpha=1.0)
        
        # Add legend
        layer_labels = {
            NodeType.INPUT: "Input (Sensory)",
            NodeType.FEATURE: "Feature",
            NodeType.CATEGORY: "Category",
            NodeType.JUDGMENT: "Judgment",
            NodeType.ABSTRACTION: "Abstraction",
            NodeType.HYPOTHESIS: "Hypothesis"
        }
        
        legend_elements = [
            mpatches.Patch(color=self.filter_flash_cmap(layers[layer_type] / len(layers)), 
                          label=label)
            for layer_type, label in layer_labels.items()
            if any(G.nodes[n]['node_type'] == layer_type for n in G.nodes())
        ]
        legend_elements.append(mpatches.Patch(color='yellow', label='Gamma Flash'))
        
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.title("FilterFlash Cognitive Model: Bottom-Up Processing",
                 fontsize=16, fontweight='bold')
        plt.text(0.5, 0.02, 
                "Node sizes represent confidence levels. Colors represent processing layers. "
                "Yellow highlights indicate gamma flashes.",
                horizontalalignment='center', fontsize=10, alpha=0.7)
        
        # Add subtitle with hamming-like encoding legend
        plt.text(0.5, 0.97, 
                f"Current Entropy State: {self._get_state_encoding(node_entropy)}",
                horizontalalignment='center', fontsize=10, 
                transform=plt.gca().transAxes)
        
        plt.axis('off')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
        return G
        
    def create_flash_filter_representation(self, 
                                          filename: Optional[str] = None) -> nx.DiGraph:
        """
        Create and visualize a network representation focused on the FlashFilter dynamics.
        
        Args:
            filename: If provided, save visualization to file
            
        Returns:
            NetworkX DiGraph with FlashFilter properties
        """
        G = nx.DiGraph()
        
        # For FlashFilter, we invert the hierarchy to show top-down flow
        # Create a layered representation based on node types
        layers = {
            NodeType.HYPOTHESIS: 0,
            NodeType.ABSTRACTION: 1,
            NodeType.JUDGMENT: 2,
            NodeType.CATEGORY: 3,
            NodeType.FEATURE: 4, 
            NodeType.INPUT: 5
        }
        
        # Add nodes with positions based on layer
        positions = {}
        node_confidence = {}
        node_entropy = {}
        node_flash_state = {}
        
        # Collect nodes by layer
        nodes_by_layer = {}
        for layer_type in layers.keys():
            nodes_by_layer[layer_type] = []
        
        # First pass: group nodes by layer
        for node_id in self.dag.graph.nodes:
            node = self.dag.graph.nodes[node_id]['data']
            layer = layers.get(node.node_type, 0)
            nodes_by_layer[node.node_type].append(node_id)
            
            # Add node to visualization graph with properties
            G.add_node(node_id, 
                      layer=layer, 
                      node_type=node.node_type,
                      instruction=node.instruction,
                      confidence=node.confidence,
                      has_flashed=node.has_flashed)
            
            # Calculate entropy
            entropy = self._calculate_entropy(node.posterior_distribution)
            node_entropy[node_id] = entropy
            node_confidence[node_id] = node.confidence
            node_flash_state[node_id] = node.has_flashed
        
        # Second pass: assign positions within each layer
        for layer_type, layer_nodes in nodes_by_layer.items():
            if not layer_nodes:
                continue
                
            layer = layers[layer_type]
            num_nodes = len(layer_nodes)
            
            for i, node_id in enumerate(layer_nodes):
                # Position nodes horizontally within their layer
                x_pos = (i + 1) / (num_nodes + 1)
                y_pos = 1.0 - (layer / (len(layers) + 1))
                positions[node_id] = (x_pos, y_pos)
        
        # Add edges but INVERT them to show top-down flow
        for edge in self.dag.graph.edges:
            target, source = edge  # Invert for FlashFilter view
            weight = self.dag.graph[source][target]['weight']
            G.add_edge(source, target, weight=weight)
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        
        # Draw edges with weights as widths and transparency
        edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
        nx.draw_networkx_edges(G, positions, width=edge_weights, alpha=0.6, 
                              edge_color='gray', arrows=True, arrowsize=15)
        
        # Draw nodes with size based on confidence and color based on layer
        node_sizes = [300 + node_confidence[n] * 300 for n in G.nodes()]
        node_colors = [layers[G.nodes[n]['node_type']] for n in G.nodes()]
        
        # Draw all nodes
        nx.draw_networkx_nodes(G, positions, node_size=node_sizes, 
                              node_color=node_colors, cmap=self.flash_filter_cmap,
                              alpha=0.8)
        
        # Highlight flashed nodes
        flashed_nodes = [n for n in G.nodes() if node_flash_state[n]]
        if flashed_nodes:
            nx.draw_networkx_nodes(G, positions, nodelist=flashed_nodes, 
                                  node_size=[node_sizes[list(G.nodes()).index(n)] + 100 
                                            for n in flashed_nodes],
                                  node_color='yellow', alpha=0.8)
        
        # Add labels with entropy and confidence
        node_labels = {node_id: f"{node_id}\nC:{node_confidence[node_id]:.2f}\nE:{node_entropy[node_id]:.2f}" 
                      for node_id in G.nodes()}
        nx.draw_networkx_labels(G, positions, labels=node_labels, font_size=8)
        
        # Add path highlight if available and if in FlashFilter mode
        if self.dag.current_path and self.dag.processing_mode == ProcessingMode.FLASH_TO_FILTER:
            # For FlashFilter, we need to reverse the path edges
            path_edges = [(self.dag.current_path[i+1], self.dag.current_path[i]) 
                         for i in range(len(self.dag.current_path)-1) 
                         if (self.dag.current_path[i+1], self.dag.current_path[i]) in G.edges()]
            if path_edges:
                nx.draw_networkx_edges(G, positions, edgelist=path_edges, 
                                      edge_color='orange', width=3.0, alpha=1.0)
        
        # Add legend
        layer_labels = {
            NodeType.INPUT: "Input (Sensory)",
            NodeType.FEATURE: "Feature",
            NodeType.CATEGORY: "Category",
            NodeType.JUDGMENT: "Judgment",
            NodeType.ABSTRACTION: "Abstraction",
            NodeType.HYPOTHESIS: "Hypothesis"
        }
        
        legend_elements = [
            mpatches.Patch(color=self.flash_filter_cmap(layers[layer_type] / len(layers)), 
                          label=label)
            for layer_type, label in layer_labels.items()
            if any(G.nodes[n]['node_type'] == layer_type for n in G.nodes())
        ]
        legend_elements.append(mpatches.Patch(color='yellow', label='Gamma Flash'))
        
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.title("FlashFilter Cognitive Model: Top-Down Processing",
                 fontsize=16, fontweight='bold')
        plt.text(0.5, 0.02, 
                "Node sizes represent confidence levels. Colors represent processing layers. "
                "Yellow highlights indicate gamma flashes.",
                horizontalalignment='center', fontsize=10, alpha=0.7)
        
        # Add subtitle with hamming-like encoding legend
        plt.text(0.5, 0.97, 
                f"Current Entropy State: {self._get_state_encoding(node_entropy)}",
                horizontalalignment='center', fontsize=10, 
                transform=plt.gca().transAxes)
        
        plt.axis('off')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
        return G
    
    def create_entropy_visualization(self, filename: Optional[str] = None) -> None:
        """
        Create a visualization focused on entropy across all nodes.
        
        Args:
            filename: If provided, save visualization to file
        """
        # Calculate entropy for each node
        entropy_values = {}
        for node_id in self.dag.graph.nodes:
            node = self.dag.graph.nodes[node_id]['data']
            entropy = self._calculate_entropy(node.posterior_distribution)
            entropy_values[node_id] = entropy
        
        # Create a spring layout for better visualization of entropy patterns
        G = self.dag.graph.copy()
        pos = nx.spring_layout(G, seed=42)
        
        plt.figure(figsize=(12, 10))
        
        # Size nodes by confidence, color by entropy
        node_sizes = [300 + G.nodes[n]['data'].confidence * 300 for n in G.nodes()]
        node_colors = [entropy_values[n] for n in G.nodes()]
        
        # Draw edges with weights as widths
        edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, 
                              edge_color='gray')
        
        # Draw nodes colored by entropy
        nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                      node_color=node_colors, cmap=self.entropy_cmap,
                                      alpha=0.9)
        
        # Add colorbar for entropy
        plt.colorbar(nodes, label='Entropy (uncertainty)', shrink=0.7)
        
        # Add labels with node IDs and entropy values
        node_labels = {node_id: f"{node_id}\nE:{entropy_values[node_id]:.2f}" 
                      for node_id in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
        
        # Add title and legend
        plt.title("Entropy Distribution in Cognitive Network",
                 fontsize=16, fontweight='bold')
        plt.text(0.5, 0.02, 
                "Node sizes represent confidence levels. Colors represent entropy (uncertainty). "
                "Higher entropy (red) indicates more uncertainty.",
                horizontalalignment='center', fontsize=10, alpha=0.7,
                transform=plt.gca().transAxes)
        
        # Add hamming-like encoding representation
        state_encoding = self._get_state_encoding(entropy_values)
        plt.text(0.5, 0.97, 
                f"System Entropy Encoding: {state_encoding}",
                horizontalalignment='center', fontsize=10,
                transform=plt.gca().transAxes)
        
        plt.axis('off')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def _calculate_entropy(self, distribution: Dict[str, float]) -> float:
        """
        Calculate Shannon entropy of a probability distribution.
        
        Args:
            distribution: Probability distribution
            
        Returns:
            Entropy value
        """
        entropy = 0.0
        for outcome, prob in distribution.items():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        return entropy
    
    def _get_state_encoding(self, entropy_values: Dict[str, float]) -> str:
        """
        Create a hamming-like encoding of system state based on entropy.
        
        Args:
            entropy_values: Dictionary of node entropy values
            
        Returns:
            String encoding of state
        """
        # Sort nodes by ID for consistent encoding
        sorted_nodes = sorted(entropy_values.keys())
        
        # Create a bit string where 1 = high entropy, 0 = low entropy
        encoding = ""
        threshold = 0.5  # Threshold for high/low entropy 
        
        for node_id in sorted_nodes:
            entropy = entropy_values[node_id]
            # Convert entropy to binary
            if entropy > threshold:
                encoding += "1"
            else:
                encoding += "0"
        
        # Format as readable blocks of 4 bits
        formatted_encoding = " ".join(encoding[i:i+4] for i in range(0, len(encoding), 4))
        
        return formatted_encoding
    
    def generate_subjective_state_representation(self) -> Dict:
        """
        Generate a representation of the system's subjective state.
        
        Returns:
            Dictionary with subjective state metrics
        """
        # Calculate key state metrics
        total_confidence = 0.0
        total_entropy = 0.0
        max_confidence_node = None
        max_confidence = -1.0
        
        node_types_count = {}
        node_states = {}
        
        for node_id in self.dag.graph.nodes:
            node = self.dag.graph.nodes[node_id]['data']
            
            # Track node type counts
            node_type = node.node_type
            node_types_count[node_type] = node_types_count.get(node_type, 0) + 1
            
            # Track confidence and entropy
            confidence = node.confidence
            entropy = self._calculate_entropy(node.posterior_distribution)
            
            total_confidence += confidence
            total_entropy += entropy
            
            # Track max confidence node
            if confidence > max_confidence:
                max_confidence = confidence
                max_confidence_node = node_id
            
            # Generate state representation for this node
            node_states[node_id] = {
                "type": str(node.node_type),
                "confidence": confidence,
                "entropy": entropy,
                "has_flashed": node.has_flashed,
                "activation_count": node.activation_count,
                # Add hamming-like encoding of node's distribution
                "state_encoding": self._encode_distribution(node.posterior_distribution)
            }
        
        # Calculate average metrics
        num_nodes = len(self.dag.graph.nodes)
        avg_confidence = total_confidence / num_nodes if num_nodes > 0 else 0
        avg_entropy = total_entropy / num_nodes if num_nodes > 0 else 0
        
        # Generate system-level metrics
        system_state = {
            "processing_mode": str(self.dag.processing_mode),
            "average_confidence": avg_confidence,
            "average_entropy": avg_entropy,
            "max_confidence_node": max_confidence_node,
            "max_confidence": max_confidence,
            "node_type_distribution": node_types_count,
            "flash_count": len(self.dag.gamma_detector.flash_history),
            "system_entropy_encoding": self._get_state_encoding({n: self._calculate_entropy(
                self.dag.graph.nodes[n]['data'].posterior_distribution) 
                for n in self.dag.graph.nodes}),
            "node_states": node_states
        }
        
        return system_state
    
    def _encode_distribution(self, distribution: Dict[str, float]) -> str:
        """
        Create a compact string encoding of a probability distribution.
        
        Args:
            distribution: Probability distribution
            
        Returns:
            String encoding
        """
        if not distribution:
            return "0000"
        
        # Sort outcomes for consistent encoding
        outcomes = sorted(distribution.keys())
        
        # Create encoding where each outcome is represented by its probability
        # quantized to 2 bits (0, 1, 2, 3)
        encoding = ""
        for outcome in outcomes:
            prob = distribution[outcome]
            # Convert probability to 2-bit representation (0-3)
            if prob < 0.25:
                encoding += "0"
            elif prob < 0.5:
                encoding += "1"
            elif prob < 0.75:
                encoding += "2"
            else:
                encoding += "3"
        
        return encoding


# Example usage
def demonstrate_visualizations(cognitive_dag):
    """
    Demonstrate the enhanced visualizations.
    
    Args:
        cognitive_dag: CognitiveDAG instance
    """
    visualizer = EnhancedCognitiveVisualizations(cognitive_dag)
    
    # Generate visualizations
    visualizer.create_filter_flash_representation(filename="enhanced_filter_flash.png")
    visualizer.create_flash_filter_representation(filename="enhanced_flash_filter.png")
    visualizer.create_entropy_visualization(filename="entropy_visualization.png")
    
    # Print subjective state representation
    subjective_state = visualizer.generate_subjective_state_representation()
    print("\n=== Subjective State Representation ===")
    for key, value in subjective_state.items():
        if key != "node_states":
            print(f"  {key}: {value}")
    
    print("\n=== Node State Encodings ===")
    for node_id, state in subjective_state["node_states"].items():
        print(f"  {node_id}: {state['state_encoding']} (Confidence: {state['confidence']:.2f}, Entropy: {state['entropy']:.2f})")
    
    print(f"\nSystem Entropy Encoding: {subjective_state['system_entropy_encoding']}")


if __name__ == "__main__":
    # This would be called from the main module
    # demonstrate_visualizations(model)
    pass
