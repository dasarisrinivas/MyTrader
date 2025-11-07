#!/bin/bash
# AWS CLI Configuration Helper for MyTrader LLM Integration

echo "=========================================="
echo "AWS CLI Configuration for MyTrader"
echo "=========================================="
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed!"
    echo "Install it with: brew install awscli"
    exit 1
fi

echo "✅ AWS CLI installed: $(aws --version)"
echo ""

# Check if already configured
if [ -f ~/.aws/credentials ]; then
    echo "⚠️  AWS credentials already exist at ~/.aws/credentials"
    echo ""
    read -p "Do you want to reconfigure? (y/N): " reconfigure
    if [[ ! $reconfigure =~ ^[Yy]$ ]]; then
        echo "Keeping existing configuration."
        echo ""
        echo "Current configuration:"
        aws configure list
        exit 0
    fi
fi

echo "=========================================="
echo "Step 1: Configure AWS Credentials"
echo "=========================================="
echo ""
echo "You'll need:"
echo "  1. AWS Access Key ID"
echo "  2. AWS Secret Access Key"
echo "  3. Default region (us-east-1 recommended for Bedrock)"
echo "  4. Output format (json recommended)"
echo ""
echo "Get your credentials from:"
echo "https://console.aws.amazon.com/ → Security credentials → Access keys"
echo ""

read -p "Press Enter to start configuration..."
echo ""

# Run AWS configure
aws configure

echo ""
echo "=========================================="
echo "Step 2: Verify Configuration"
echo "=========================================="
echo ""

aws configure list

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ AWS CLI configured successfully!"
else
    echo ""
    echo "❌ Configuration failed. Please try again."
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 3: Test AWS Connection"
echo "=========================================="
echo ""

echo "Testing AWS connection..."
aws sts get-caller-identity 2>/dev/null

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ AWS connection successful!"
else
    echo ""
    echo "⚠️  Could not verify AWS connection."
    echo "Please check your credentials and try again."
fi

echo ""
echo "=========================================="
echo "Step 4: AWS Bedrock Model Access"
echo "=========================================="
echo ""
echo "✅ Good news! Model access is now automatic!"
echo ""
echo "📝 All Bedrock serverless models are automatically enabled"
echo "   No manual approval needed!"
echo ""
echo "Available models:"
echo "   ✓ Claude 3 Sonnet (anthropic.claude-3-sonnet-20240229-v1:0)"
echo "   ✓ Claude 3 Haiku (anthropic.claude-3-haiku-20240307-v1:0)"
echo "   ✓ Claude 3.5 Sonnet (anthropic.claude-3-5-sonnet-20240620-v1:0)"
echo ""
echo "⚠️  Note: First-time Anthropic users may need to submit use case details"
echo "   Visit: https://console.aws.amazon.com/bedrock/home#/model-catalog"
echo ""
echo "🔒 To restrict access (optional):"
echo "   Use IAM policies and Service Control Policies as needed"
echo ""
echo "📚 Documentation:"
echo "   https://docs.aws.amazon.com/bedrock/latest/userguide/"
echo ""

echo "=========================================="
echo "Step 5: Enable LLM in MyTrader"
echo "=========================================="
echo ""
echo "Edit config.yaml and set:"
echo ""
echo "llm:"
echo "  enabled: true"
echo "  model_id: \"anthropic.claude-3-sonnet-20240229-v1:0\""
echo "  region_name: \"us-east-1\""
echo "  min_confidence_threshold: 0.7"
echo ""

echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. ✅ AWS CLI configured"
echo "  2. ✅ Bedrock models automatically enabled"
echo "  3. Enable LLM in config.yaml (see below)"
echo "  4. Test: python3 example_llm_integration.py"
echo ""
echo "For more info, see: LLM_INTEGRATION.md"
echo ""
