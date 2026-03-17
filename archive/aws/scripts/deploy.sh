#!/bin/bash
#
# Deploy Trading Bot AWS Infrastructure
#
# Usage:
#   ./deploy.sh [environment] [region]
#
# Examples:
#   ./deploy.sh prod us-east-1
#   ./deploy.sh dev us-west-2
#

set -e

# Configuration
PROJECT_NAME="trading-bot"
ENVIRONMENT="${1:-prod}"
REGION="${2:-us-east-1}"
STACK_NAME="${PROJECT_NAME}-${ENVIRONMENT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Trading Bot AWS Deployment${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Project:     ${PROJECT_NAME}"
echo "Environment: ${ENVIRONMENT}"
echo "Region:      ${REGION}"
echo "Stack:       ${STACK_NAME}"
echo ""

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI is not installed${NC}"
    exit 1
fi

# Check AWS credentials
echo -e "${YELLOW}Checking AWS credentials...${NC}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo -e "${RED}Error: AWS credentials not configured${NC}"
    exit 1
fi
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
    
    # Create zip file
    cd "$lambda_dir"
    zip -r "/tmp/${lambda_name}.zip" . -x "*.pyc" -x "__pycache__/*" > /dev/null
    
    # Upload to S3
    aws s3 cp "/tmp/${lambda_name}.zip" \
        "s3://${ARTIFACT_BUCKET}/lambda-code/${lambda_name}.zip" \
        --region "${REGION}"
done

# Create Lambda layer (you would build this separately with dependencies)
echo -e "\n${YELLOW}Note: Lambda layer must be built separately with dependencies${NC}"
echo "Run: pip install pyarrow pandas numpy -t python/lib/python3.12/site-packages/"
echo "Then: zip -r common-deps.zip python/"

# Package CloudFormation templates
echo -e "\n${YELLOW}Packaging CloudFormation templates...${NC}"
aws cloudformation package \
    --template-file "${CF_DIR}/main.yaml" \
    --s3-bucket "${ARTIFACT_BUCKET}" \
    --s3-prefix "cloudformation" \
    --output-template-file "/tmp/packaged-template.yaml" \
    --region "${REGION}"

# Deploy main stack
echo -e "\n${YELLOW}Deploying CloudFormation stack: ${STACK_NAME}${NC}"
aws cloudformation deploy \
    --template-file "/tmp/packaged-template.yaml" \
    --stack-name "${STACK_NAME}" \
    --parameter-overrides \
        Environment="${ENVIRONMENT}" \
        ProjectName="${PROJECT_NAME}" \
        BucketName="${PROJECT_NAME}-data" \
        ArtifactsBucket="${ARTIFACT_BUCKET}" \
    --capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND \
    --region "${REGION}" \
    --tags \
        Project="${PROJECT_NAME}" \
        Environment="${ENVIRONMENT}"

# Get stack outputs
echo -e "\n${YELLOW}Getting stack outputs...${NC}"
aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --query "Stacks[0].Outputs" \
    --output table \
    --region "${REGION}"

# Initialize S3 folder structure
echo -e "\n${YELLOW}Initializing S3 folder structure...${NC}"
DATA_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --query "Stacks[0].Outputs[?OutputKey=='S3BucketName'].OutputValue" \
    --output text \
    --region "${REGION}")

if [ -n "$DATA_BUCKET" ]; then
    python3 "${SCRIPT_DIR}/init_s3_folders.py" --bucket "${DATA_BUCKET}" --region "${REGION}"
fi

# Output configuration for local bot
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Add the following to your local config.yaml:"
echo ""
echo "aws_agents:"
echo "  enabled: true"
echo "  region_name: ${REGION}"
echo "  s3_bucket: ${DATA_BUCKET}"

# Get agent IDs
AGENT1_ID=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --query "Stacks[0].Outputs[?OutputKey=='Agent1Id'].OutputValue" \
    --output text \
    --region "${REGION}" 2>/dev/null || echo "")

AGENT2_ID=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --query "Stacks[0].Outputs[?OutputKey=='Agent2Id'].OutputValue" \
    --output text \
    --region "${REGION}" 2>/dev/null || echo "")

AGENT3_ID=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --query "Stacks[0].Outputs[?OutputKey=='Agent3Id'].OutputValue" \
    --output text \
    --region "${REGION}" 2>/dev/null || echo "")

AGENT4_ID=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --query "Stacks[0].Outputs[?OutputKey=='Agent4Id'].OutputValue" \
    --output text \
    --region "${REGION}" 2>/dev/null || echo "")

echo "  agent1_id: ${AGENT1_ID}"
echo "  agent2_id: ${AGENT2_ID}"
echo "  agent3_id: ${AGENT3_ID}"
echo "  agent4_id: ${AGENT4_ID}"
echo ""
