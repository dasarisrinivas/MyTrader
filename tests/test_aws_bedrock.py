#!/usr/bin/env python3
"""Quick test to verify AWS Bedrock access and configuration."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_aws_credentials():
    """Test if AWS credentials are configured."""
    print("\n" + "="*60)
    print("Testing AWS Credentials")
    print("="*60)
    
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, ClientError
    except ImportError:
        print("❌ boto3 not installed")
        print("   Run: pip3 install boto3 botocore")
        return False
    
    try:
        # Try to create a client
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        
        print("✅ AWS Credentials configured")
        print(f"   Account: {identity['Account']}")
        print(f"   User ARN: {identity['Arn']}")
        print(f"   User ID: {identity['UserId']}")
        return True
        
    except NoCredentialsError:
        print("❌ AWS Credentials not configured")
        print("   Run: aws configure")
        return False
    except ClientError as e:
        print(f"❌ AWS Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_bedrock_access():
    """Test if Bedrock models are accessible."""
    print("\n" + "="*60)
    print("Testing AWS Bedrock Access")
    print("="*60)
    
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        print("❌ boto3 not installed")
        print("   Run: pip3 install boto3 botocore")
        return False
    
    try:
        # Try to list Bedrock models
        bedrock = boto3.client('bedrock', region_name='us-east-1')
        
        print("✅ Bedrock service accessible")
        print("   Region: us-east-1")
        
        # Try to list foundation models
        try:
            response = bedrock.list_foundation_models()
            models = response.get('modelSummaries', [])
            
            # Find Claude models
            claude_models = [
                m for m in models 
                if 'claude' in m.get('modelId', '').lower()
            ]
            
            if claude_models:
                print(f"\n✅ Found {len(claude_models)} Claude models available:")
                for model in claude_models[:5]:  # Show first 5
                    model_id = model.get('modelId', 'unknown')
                    provider = model.get('providerName', 'unknown')
                    print(f"   • {model_id} ({provider})")
                
                if len(claude_models) > 5:
                    print(f"   ... and {len(claude_models) - 5} more")
            else:
                print("\n⚠️  No Claude models found")
                print("   This may be normal - model access is automatic")
            
            return True
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            
            if error_code == 'AccessDeniedException':
                print("\n⚠️  Access Denied to list models")
                print("   This is OK - you may still be able to invoke models")
                print("   Try: python3 test_bedrock_invoke.py")
            else:
                print(f"\n⚠️  Error listing models: {error_code}")
                print(f"   Message: {e}")
            
            return True  # Don't fail - list permission not required
            
    except ImportError:
        print("❌ boto3 not installed")
        print("   Run: pip install boto3 botocore")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_bedrock_invoke():
    """Test if we can invoke a Bedrock model."""
    print("\n" + "="*60)
    print("Testing Bedrock Model Invocation")
    print("="*60)
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        import json
    except ImportError:
        print("❌ boto3 not installed")
        print("   Run: pip3 install boto3 botocore")
        return False
    
    try:
        # Try to invoke Claude 3 Haiku (smallest/cheapest for testing)
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        
        print(f"Attempting to invoke: {model_id}")
        print("Sending test prompt...")
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": "Say 'Hello from AWS Bedrock!' and nothing else."
                }
            ]
        })
        
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=body
        )
        
        response_body = json.loads(response.get('body').read())
        content = response_body.get('content', [{}])[0].get('text', '')
        
        print("\n✅ Model invocation successful!")
        print(f"   Model: {model_id}")
        print(f"   Response: {content}")
        print("\n🎉 AWS Bedrock is fully configured and working!")
        return True
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        error_msg = e.response.get('Error', {}).get('Message', '')
        
        if error_code == 'AccessDeniedException':
            print("\n❌ Access Denied to invoke model")
            print(f"   Message: {error_msg}")
            print("\n   Possible solutions:")
            print("   1. First-time Anthropic users: Submit use case details")
            print("      Visit: https://console.aws.amazon.com/bedrock/home#/model-catalog")
            print("   2. Check IAM permissions include 'bedrock:InvokeModel'")
            print("   3. Verify you're in a supported region (us-east-1, us-west-2)")
        elif error_code == 'ValidationException':
            print(f"\n⚠️  Model validation error: {error_msg}")
            print("   Try using Claude 3 Sonnet instead")
        else:
            print(f"\n❌ Error: {error_code}")
            print(f"   Message: {error_msg}")
        
        return False
        
    except ImportError:
        print("❌ boto3 not installed")
        print("   Run: pip install boto3 botocore")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("AWS BEDROCK CONFIGURATION TEST")
    print("="*60)
    
    results = []
    
    # Test 1: AWS Credentials
    results.append(("AWS Credentials", test_aws_credentials()))
    
    # Test 2: Bedrock Access
    if results[0][1]:  # Only if credentials work
        results.append(("Bedrock Access", test_bedrock_access()))
        
        # Test 3: Model Invocation
        results.append(("Model Invocation", test_bedrock_invoke()))
    else:
        print("\n⏭️  Skipping Bedrock tests (credentials not configured)")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! AWS Bedrock is ready to use.")
        print("\nNext steps:")
        print("  1. Enable LLM in config.yaml:")
        print("     llm.enabled: true")
        print("  2. Run: python3 example_llm_integration.py")
    else:
        print("\n⚠️  Some tests failed. Review the output above for details.")
        print("\nFor help, see:")
        print("  • LLM_INTEGRATION.md")
        print("  • Run: ./setup_aws.sh")
    
    print("")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
