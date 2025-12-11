#!/bin/bash

# Two-Stack Deployment Script for Trading Bot AWS Infrastructure
# Stack 1: Foundation (S3, IAM, OpenSearch Collection)
# Stack 2: Services (Knowledge Base, Lambda, Agents, Step Functions)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
ENVIRONMENT="${1:-dev}"
PROJECT_NAME="trading-bot"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
STACK1_NAME="${PROJECT_NAME}-${ENVIRONMENT}-foundation"
STACK2_NAME="${PROJECT_NAME}-${ENVIRONMENT}-services"

echo -e "${GREEN}========================================"
echo "Trading Bot Two-Stack Deployment"
echo "========================================${NC}"
echo ""
echo "Project:     ${PROJECT_NAME}"
echo "Environment: ${ENVIRONMENT}"
echo "Region:      ${REGION}"
echo "Stack 1:     ${STACK1_NAME}"
echo "Stack 2:     ${STACK2_NAME}"
echo ""

# Check AWS credentials
echo -e "${YELLOW}Checking AWS credentials...${NC}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo -e "${GREEN}AWS Account: ${AWS_ACCOUNT_ID}${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF_DIR="$(dirname "$SCRIPT_DIR")/cloudformation"
LAMBDA_DIR="$(dirname "$SCRIPT_DIR")/lambda"

# Create S3 bucket for deployment artifacts
ARTIFACT_BUCKET="${PROJECT_NAME}-artifacts-${AWS_ACCOUNT_ID}"
echo -e "\n${YELLOW}Creating artifact bucket: ${ARTIFACT_BUCKET}${NC}"
aws s3 mb "s3://${ARTIFACT_BUCKET}" --region "${REGION}" 2>/dev/null || true

# Package Lambda functions
echo -e "\n${YELLOW}Packaging Lambda functions...${NC}"
for lambda_dir in "${LAMBDA_DIR}"/*/; do
    lambda_name=$(basename "$lambda_dir")
    echo "  Packaging: ${lambda_name}"
    cd "$lambda_dir"
    zip -r "/tmp/${lambda_name}.zip" . -x "*.pyc" -x "__pycache__/*" > /dev/null
    aws s3 cp "/tmp/${lambda_name}.zip" \
        "s3://${ARTIFACT_BUCKET}/lambda-code/${lambda_name}.zip" \
        --region "${REGION}"
done

# Check if layer exists
echo -e "\n${YELLOW}Checking Lambda layer...${NC}"
if ! aws s3 ls "s3://${ARTIFACT_BUCKET}/lambda-layers/common-deps.zip" 2>/dev/null; then
    echo -e "${RED}Lambda layer not found. Please build and upload it first:${NC}"
    echo "  mkdir -p /tmp/lambda-layer/python"
    echo "  pip3 install pyarrow pandas numpy -t /tmp/lambda-layer/python"
    echo "  cd /tmp/lambda-layer && zip -r common-deps.zip python"
    echo "  aws s3 cp common-deps.zip s3://${ARTIFACT_BUCKET}/lambda-layers/common-deps.zip"
    echo ""
    read -p "Press Enter after uploading the layer, or Ctrl+C to cancel..."
fi

# Function to deploy a stack
deploy_stack() {
    local template=$1
    local stack_name=$2
    local extra_params=$3
    
    echo -e "\n${YELLOW}Deploying: ${stack_name}${NC}"
    
    aws cloudformation deploy \
        --template-file "${template}" \
        --stack-name "${stack_name}" \
        --parameter-overrides \
            Environment="${ENVIRONMENT}" \
            ProjectName="${PROJECT_NAME}" \
            ArtifactsBucket="${ARTIFACT_BUCKET}" \
            ${extra_params} \
        --capabilities CAPABILITY_NAMED_IAM \
        --region "${REGION}" \
        --tags \
            Project="${PROJECT_NAME}" \
            Environment="${ENVIRONMENT}"
}

# Check command line argument
case "${2:-}" in
    "stack1")
        echo -e "\n${GREEN}=== Deploying Stack 1: Foundation ===${NC}"
        deploy_stack "${CF_DIR}/stack1-foundation.yaml" "${STACK1_NAME}"
        
        echo -e "\n${GREEN}========================================${NC}"
        echo -e "${GREEN}Stack 1 deployed successfully!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo -e "${YELLOW}IMPORTANT: Wait 10-15 minutes for OpenSearch data access policies to propagate.${NC}"
        echo ""
        echo "Then create the vector index manually:"
        echo "  1. Go to AWS OpenSearch Serverless console"
        echo "  2. Select collection: ${PROJECT_NAME}-${ENVIRONMENT}-vectors"
        echo "  3. Create index: ${PROJECT_NAME}-trading-index"
        echo "     - Vector field: embedding (dimension 1024, HNSW, faiss)"
        echo "     - Text field: text"
        echo "     - Metadata field: metadata"
        echo ""
        echo "Or run the index creation script:"
        echo "  python3 ${SCRIPT_DIR}/create_opensearch_index.py --environment ${ENVIRONMENT}"
        echo ""
        echo "After index is created, deploy Stack 2:"
        echo "  ./deploy_two_stacks.sh ${ENVIRONMENT} stack2"
        ;;
        
    "stack2")
        echo -e "\n${GREEN}=== Deploying Stack 2: Services ===${NC}"
        
        # Check if Stack 1 exists
        if ! aws cloudformation describe-stacks --stack-name "${STACK1_NAME}" --region "${REGION}" &>/dev/null; then
            echo -e "${RED}Error: Stack 1 (${STACK1_NAME}) must be deployed first!${NC}"
            echo "Run: ./deploy_two_stacks.sh ${ENVIRONMENT} stack1"
            exit 1
        fi
        
        # Check if index exists (optional - just a warning)
        echo -e "${YELLOW}Note: Make sure the OpenSearch index exists before continuing.${NC}"
        read -p "Press Enter to continue deployment, or Ctrl+C to cancel..."
        
        deploy_stack "${CF_DIR}/stack2-services.yaml" "${STACK2_NAME}" \
            "VectorIndexName=${PROJECT_NAME}-trading-index"
        
        echo -e "\n${GREEN}========================================${NC}"
        echo -e "${GREEN}Stack 2 deployed successfully!${NC}"
        echo -e "${GREEN}========================================${NC}"
        
        # Show outputs
        echo -e "\n${YELLOW}Stack Outputs:${NC}"
        aws cloudformation describe-stacks \
            --stack-name "${STACK2_NAME}" \
            --query "Stacks[0].Outputs" \
            --output table \
            --region "${REGION}"
        ;;
        
    "all")
        echo -e "\n${GREEN}=== Deploying Both Stacks ===${NC}"
        echo -e "${YELLOW}Warning: This will deploy Stack 1, wait, then deploy Stack 2${NC}"
        
        # Deploy Stack 1
        deploy_stack "${CF_DIR}/stack1-foundation.yaml" "${STACK1_NAME}"
        
        echo -e "\n${YELLOW}Stack 1 complete. Waiting 10 minutes for policy propagation...${NC}"
        for i in {10..1}; do
            echo -ne "  ${i} minutes remaining...\r"
            sleep 60
        done
        echo ""
        
        # Create index
        echo -e "\n${YELLOW}Creating OpenSearch index...${NC}"
        python3 "${SCRIPT_DIR}/create_opensearch_index.py" --environment "${ENVIRONMENT}" || {
            echo -e "${RED}Index creation failed. Please create manually and run stack2.${NC}"
            exit 1
        }
        
        # Deploy Stack 2
        deploy_stack "${CF_DIR}/stack2-services.yaml" "${STACK2_NAME}" \
            "VectorIndexName=${PROJECT_NAME}-trading-index"
        
        echo -e "\n${GREEN}========================================${NC}"
        echo -e "${GREEN}Full Deployment Complete!${NC}"
        echo -e "${GREEN}========================================${NC}"
        ;;
        
    "status")
        echo -e "\n${YELLOW}Stack Status:${NC}"
        echo ""
        echo "Stack 1 (${STACK1_NAME}):"
        aws cloudformation describe-stacks --stack-name "${STACK1_NAME}" \
            --query "Stacks[0].StackStatus" --output text --region "${REGION}" 2>/dev/null || echo "  NOT DEPLOYED"
        echo ""
        echo "Stack 2 (${STACK2_NAME}):"
        aws cloudformation describe-stacks --stack-name "${STACK2_NAME}" \
            --query "Stacks[0].StackStatus" --output text --region "${REGION}" 2>/dev/null || echo "  NOT DEPLOYED"
        ;;
        
    "delete")
        echo -e "\n${RED}=== Deleting Stacks ===${NC}"
        echo -e "${YELLOW}This will delete both stacks. Are you sure?${NC}"
        read -p "Type 'yes' to confirm: " confirm
        if [ "$confirm" = "yes" ]; then
            echo "Deleting Stack 2..."
            aws cloudformation delete-stack --stack-name "${STACK2_NAME}" --region "${REGION}" 2>/dev/null || true
            aws cloudformation wait stack-delete-complete --stack-name "${STACK2_NAME}" --region "${REGION}" 2>/dev/null || true
            
            echo "Deleting Stack 1..."
            aws cloudformation delete-stack --stack-name "${STACK1_NAME}" --region "${REGION}" 2>/dev/null || true
            aws cloudformation wait stack-delete-complete --stack-name "${STACK1_NAME}" --region "${REGION}" 2>/dev/null || true
            
            echo -e "${GREEN}Stacks deleted.${NC}"
        else
            echo "Cancelled."
        fi
        ;;
        
    *)
        echo "Usage: $0 <environment> <command>"
        echo ""
        echo "Commands:"
        echo "  stack1  - Deploy Stack 1 (Foundation: S3, IAM, OpenSearch)"
        echo "  stack2  - Deploy Stack 2 (Services: KB, Lambda, Agents)"
        echo "  all     - Deploy both stacks with wait"
        echo "  status  - Check stack status"
        echo "  delete  - Delete both stacks"
        echo ""
        echo "Examples:"
        echo "  $0 dev stack1    # Deploy foundation first"
        echo "  $0 dev stack2    # Deploy services after waiting"
        echo "  $0 dev all       # Deploy everything with automatic wait"
        echo "  $0 dev status    # Check deployment status"
        ;;
esac
