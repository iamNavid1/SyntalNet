#!/bin/bash

# =============================================================================
# Script to train ablation variants of GLRX multimodal fusion:
#   - UniformAvgFusion (uniform_avg) - Uniform averaging baseline
#   - ConcatMLP (concat_mlp) - Concatenation + MLP projection
#   - GatedSumOnly (gated_sum) - Gated sum only
#   - PairwiseOnly (pairwise) - Pairwise only
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
LOG_FILE="logs/train_glrx_ablations_${TIMESTAMP}.txt"
mkdir -p logs

exec > >(tee -a "$LOG_FILE") 2>&1

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}Training GLRX ablation variants:${NC}"
echo -e "${BLUE}Log file: $LOG_FILE${NC}"
echo -e "${BLUE}=============================================================================${NC}"

# Config files
CONFIGS=(
    "configs/EXPT_GLRX_uniform_avg.yaml"
    "configs/EXPT_GLRX_concat_mlp.yaml"
    "configs/EXPT_GLRX_gated_sum.yaml"
    "configs/EXPT_GLRX_pairwise.yaml"
)

# Variant names
EXPERIMENTS=(
    "UniformAvgFusion"
    "ConcatMLP"
    "GatedSumOnly"
    "PairwiseOnly"
)

# Check if train.py exists
if [ ! -f "train.py" ]; then
    echo -e "${RED}Error: train.py not found in current directory${NC}"
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
echo "Starting training at: $(date)"
echo ""

# Track time
START_TIME=$(date +%s)

# Train variants
for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    exp_name="${EXPERIMENTS[$i]}"
    
    echo -e "${YELLOW}=============================================================================${NC}"
    echo -e "${YELLOW}[$((i+1))/${#CONFIGS[@]}] Starting: $exp_name${NC}"
    echo -e "${YELLOW}Config: $config${NC}"
    echo -e "${YELLOW}=============================================================================${NC}"
    echo ""
    
    # Run the training
    if python train.py --config "$config" --resume; then
        echo ""
        echo -e "${GREEN}✓ Experiment $exp_name completed successfully${NC}"
        echo ""
    else
        echo ""
        echo -e "${RED}✗ Experiment $exp_name failed!${NC}"
        echo -e "${RED}Stopping execution.${NC}"
        exit 1
    fi
done

# Elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo -e "${GREEN}=============================================================================${NC}"
echo -e "${GREEN}All experiments completed successfully!${NC}"
echo -e "${GREEN}=============================================================================${NC}"
echo ""
echo "Ending time: $(date)"
echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo -e "${BLUE}Results saved in:${NC}"
for i in "${!CONFIGS[@]}"; do
    exp_name="${EXPERIMENTS[$i]}"
    config="${CONFIGS[$i]}"
    ckpt_dir=$(grep "checkpoint_dir:" "$config" | awk '{print $2}')
    echo "  - $exp_name: $ckpt_dir"
done
echo ""
echo -e "${BLUE}Full log saved to: $LOG_FILE${NC}"
echo ""

