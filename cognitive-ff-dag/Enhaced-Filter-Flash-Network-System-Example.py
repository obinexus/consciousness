"""
Enhanced Filter-Flash Cognitive Model Implementation Example

This example demonstrates the application of the Bayesian DAG-based 
cognitive model with enhanced visualizations and state representations.
"""

import sys
import os
from typing import Dict, List, Tuple, Optional

# Import core cognitive system components
# Assuming these files are in the same directory
from cognitive_system import (
    CognitiveDAG, CognitiveNode, NodeType, 
    InstructionType, ProcessingMode, 
    build_color_classification_model
)

# Import enhanced visualizations
from cognitive_visualizations import EnhancedCognitiveVisualizations


def run_enhanced_example():
    """Run an enhanced example of the filter-flash cognitive system."""
    print("\n=== Running Enhanced Filter-Flash Cognitive System Example ===\n")
    
    # Build the color classification model
    model = build_color_classification_model()
    print("Model built successfully with the following node types:")
    
    # Count node types
    node_type_counts = {}
    for node_id in model.graph.nodes:
        node = model.graph.nodes[node_id]['data']
        node_type = node.node_type
        node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
    
    for node_type, count in node_type_counts.items():
        print(f"  {node_type.name}: {count} nodes")
    
    # Step 1: Bottom-up processing (FilterFlash)
    print("\n=== Step 1: FilterFlash Classification (Bottom-Up) ===")
    print("Processing high red, low green, low blue input...")
    
    # Process inputs sequentially
    flashed_nodes = model.filter_to_flash(
        {"high": 0.9, "low": 0.1}, ["r_input"]
    )
    flashed_nodes += model.filter_to_flash(
        {"high": 0.2, "low": 0.8}, ["g_input"]
    )
    flashed_nodes += model.filter_to_flash(
        {"high": 0.1, "low": 0.9}, ["b_input"]
    )
    
    print(f"Flashed nodes during FilterFlash: {flashed_nodes}")
    
    # Create enhanced visualizations
    visualizer = EnhancedCognitiveVisualizations(model)
    
    # Generate FilterFlash representation
    print("\nGenerating FilterFlash visualization...")
    visualizer.create_filter_flash_representation(filename="enhanced_filter_flash.png")
    print("Visualization saved to 'enhanced_filter_flash.png'")
    
    # Print node states with confidence and entropy
    print("\nNode States after FilterFlash processing:")
    for node_id in model.graph.nodes:
        node = model.graph.nodes[node_id]['data']
        entropy = visualizer._calculate_entropy(node.posterior_distribution)
        print(f"  {node_id}: Confidence={node.confidence:.4f}, Entropy={entropy:.4f}")
        if node.has_flashed:
            print(f"    ** GAMMA FLASH occurred at this node **")
    
    # Step 2: Top-down processing (FlashFilter)
    print("\n=== Step 2: FlashFilter Verification (Top-Down) ===")
    print("Testing hypothesis that this is a red color...")
    
    # Set up a hypothesis that this is a red color
    hypothesis = {
        "red": 0.7,
        "green": 0.1,
        "blue": 0.1, 
        "yellow": 0.1
    }
    
    # Execute top-down processing
    flashed_nodes = model.flash_to_filter("color_hypothesis", hypothesis)
    print(f"Flashed nodes during FlashFilter: {flashed_nodes}")
    
    # Generate FlashFilter representation
    print("\nGenerating FlashFilter visualization...")
    visualizer.create_flash_filter_representation(filename="enhanced_flash_filter.png")
    print("Visualization saved to 'enhanced_flash_filter.png'")
    
    # Step 3: Provide feedback and update model
    print("\n=== Step 3: Learning from Feedback ===")
    
    feedback_data = {
        "correct": 0.9,
        "incorrect": 0.1
    }
    
    print("Providing positive feedback (90% correct)...")
    model.update_from_feedback(feedback_data, ["feedback"])
    
    # Generate entropy visualization
    print("\nGenerating entropy visualization to show uncertainty patterns...")
    visualizer.create_entropy_visualization(filename="entropy_visualization.png")
    print("Entropy visualization saved to 'entropy_visualization.png'")
    
    # Generate detailed subjective state representation
    print("\n=== Subjective State Representation ===")
    subjective_state = visualizer.generate_subjective_state_representation()
    
    # Print system-level metrics
    for key, value in subjective_state.items():
        if key != "node_states":
            print(f"  {key}: {value}")
    
    # Print node encodings
    print("\n=== Node State Encodings (Hamming-style) ===")
    for node_id, state in subjective_state["node_states"].items():
        if state["confidence"] > 0:  # Only show active nodes
            print(f"  {node_id}: {state['state_encoding']} (Conf: {state['confidence']:.2f}, Entropy: {state['entropy']:.2f})")
    
    print(f"\nSystem Entropy Encoding: {subjective_state['system_entropy_encoding']}")
    
    # Print flash and execution summaries
    flash_summary = model.gamma_detector.get_flash_summary()
    execution_summary = model.get_execution_summary()
    
    print("\n=== Gamma Flash Summary ===")
    for key, value in flash_summary.items():
        print(f"  {key}: {value}")
    
    print("\n=== Execution Summary ===")
    for key, value in execution_summary.items():
        print(f"  {key}: {value}")
    
    print("\nEnhanced Filter-Flash Cognitive Model demonstration complete!")


if __name__ == "__main__":
    run_enhanced_example()
