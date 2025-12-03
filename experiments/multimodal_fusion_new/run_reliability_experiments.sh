#!/bin/bash

# ============================================================================= 
# Script to run reliability-switch experiments for multimodal fusion variants
# Trains and evaluates: GLR-X, UniformAvg, ConcatMLP
# =============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# Output colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/reliability_experiments_${TIMESTAMP}.txt"
mkdir -p logs

exec > >(tee -a "$LOG_FILE") 2>&1

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}Reliability-Switch Multimodal Fusion Experiments${NC}"
echo -e "${BLUE}Log file: $LOG_FILE${NC}"
echo -e "${BLUE}=============================================================================${NC}"

# Configuration files
CONFIGS=(
    "experiments/multimodal_fusion_new/configs/reliability_glrx.yaml"
    "experiments/multimodal_fusion_new/configs/reliability_uniform_avg.yaml"
    "experiments/multimodal_fusion_new/configs/reliability_concat_mlp.yaml"
)

# Experiment names
EXPERIMENTS=(
    "GLR-X"
    "UniformAvg"
    "ConcatMLP"
)

# Check if runner exists
RUNNER="experiments/multimodal_fusion_new/runner.py"
if [ ! -f "$RUNNER" ]; then
    echo -e "${RED}Error: runner.py not found at $RUNNER${NC}"
    exit 1
fi

# Check if all config files exist
for config in "${CONFIGS[@]}"; do
    if [ ! -f "$config" ]; then
        echo -e "${RED}Error: Config file not found: $config${NC}"
        exit 1
    fi
done

echo ""
echo "Total variants: ${#CONFIGS[@]}"
echo "Starting experiments at: $(date)"
echo ""

# Track time
START_TIME=$(date +%s)

# Run experiments
for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    exp_name="${EXPERIMENTS[$i]}"
    
    echo -e "${YELLOW}=============================================================================${NC}"
    echo -e "${YELLOW}[$((i+1))/${#CONFIGS[@]}] Starting: $exp_name${NC}"
    echo -e "${YELLOW}Config: $config${NC}"
    echo -e "${YELLOW}=============================================================================${NC}"
    echo ""
    
    # Run the experiment (train + eval)
    if python "$RUNNER" --config "$config" --mode eval --device cuda:0; then
        echo ""
        echo -e "${GREEN}✓ Experiment $exp_name completed successfully${NC}"
        echo ""
    else
        echo ""
        echo -e "${RED}✗ Experiment $exp_name failed!${NC}"
        echo -e "${RED}Continuing to next experiment...${NC}"
        echo ""
    fi
done

# Elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo -e "${GREEN}=============================================================================${NC}"
echo -e "${GREEN}All experiments completed!${NC}"
echo -e "${GREEN}=============================================================================${NC}"
echo ""
echo "Ending time: $(date)"
echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo -e "${BLUE}Results saved in:${NC}"
echo "  - experiments/multimodal_fusion_reliability_results/"
echo ""
echo -e "${BLUE}Full log saved to: $LOG_FILE${NC}"
echo ""

