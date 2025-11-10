#!/bin/bash

# AI Pronunciation Trainer - Easy Deployment Script
# This script helps you deploy to AWS Lambda quickly

set -e  # Exit on error

echo "=========================================="
echo "AI Pronunciation Trainer - Deployment"
echo "=========================================="
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install it first:"
    echo "   https://aws.amazon.com/cli/"
    exit 1
fi

# Check if SAM CLI is installed
if ! command -v sam &> /dev/null; then
    echo "❌ AWS SAM CLI not found. Please install it first:"
    echo "   https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html"
    exit 1
fi

echo "✅ AWS CLI and SAM CLI are installed"
echo ""

# Check for API key in environment
if [ -z "$WHISPER_API_KEY" ]; then
    echo "⚠️  WHISPER_API_KEY not set in environment"
    echo ""
    echo "Get your OpenAI API key here: https://platform.openai.com/api-keys"
    echo ""
    read -p "Enter your OpenAI API key: " WHISPER_API_KEY
    
    if [ -z "$WHISPER_API_KEY" ]; then
        echo "❌ API key cannot be empty. Exiting."
        exit 1
    fi

    read -p "Enter your OpenAI API base: " OPENAI_API_BASE
    
    if [ -z "$OPENAI_API_BASE" ]; then
        echo "❌ API base cannot be empty. Exiting."
        exit 1
    fi
else
    echo "✅ Using OpenAI API key from environment"
fi

echo ""
echo "=========================================="
echo "Building application..."
echo "=========================================="
sam build

if [ $? -eq 0 ]; then
    echo "✅ Build successful"
else
    echo "❌ Build failed. Please check the errors above."
    exit 1
fi

echo ""
echo "=========================================="
echo "Deploying to AWS..."
echo "=========================================="
echo ""
echo "You will be asked a few questions:"
echo "- Stack Name: pronunciation-trainer (or your choice)"
echo "- AWS Region: us-east-1 (or your preferred region)"
echo "- Confirm changes: Y"
echo "- Allow SAM CLI IAM role creation: Y"
echo "- Save arguments to configuration: Y"
echo ""
read -p "Press Enter to continue with deployment..."

sam deploy \
    --parameter-overrides \
        WhisperApiKey="$WHISPER_API_KEY" \
        OpenaiApiBase="$OPENAI_API_BASE"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Deployment successful!"
    echo "=========================================="
    echo ""
    echo "Your API is now live!"
    echo ""
    echo "Next steps:"
    echo "1. Copy the API URL from the output above"
    echo "2. Update your frontend to use this URL"
    echo "3. Test the endpoint with the example in QUICK_START.md"
    echo ""
    echo "Monitor your application:"
    echo "- CloudWatch Logs: https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#logsV2:log-groups"
    echo "- Lambda Functions: https://console.aws.amazon.com/lambda/home?region=us-east-1#/functions"
    echo ""
else
    echo ""
    echo "❌ Deployment failed. Please check the errors above."
    exit 1
fi

