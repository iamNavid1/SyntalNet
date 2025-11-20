#!/bin/bash
# Example script to run GLRX stress test evaluation
#
# This script demonstrates how to run the complete GLRX evaluation pipeline
# with different configurations.

# Configuration
CONFIG_PATH="configs/SyntalNet.yaml"
CHECKPOINT_DIR="checkpoints/glrx_variants"
OUTPUT_DIR="results/multimodal_fusion"

# Example 1: Run all variants with allocation tracking
echo "Running GLRX experiments with all variants..."
python experiments/multimodal_fusion/runner.py \
    --config "$CONFIG_PATH" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "$OUTPUT_DIR"

# Example 2: Run specific variants only
echo ""
echo "Running GLRX experiments with specific variants..."
python experiments/multimodal_fusion/runner.py \
    --config "$CONFIG_PATH" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "${OUTPUT_DIR}_glrx_only" \
    --variants glrx uniform_avg

# Example 3: Run without allocation tracking
echo ""
echo "Running GLRX experiments without allocation tracking..."
python experiments/multimodal_fusion/runner.py \
    --config "$CONFIG_PATH" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "${OUTPUT_DIR}_no_alloc" \
    --no-allocation

# Example 4: Run on specific GPU
echo ""
echo "Running GLRX experiments on GPU 0..."
python experiments/multimodal_fusion/runner.py \
    --config "$CONFIG_PATH" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "${OUTPUT_DIR}_gpu0" \
    --device cuda:0

echo ""
echo "All examples complete!"

