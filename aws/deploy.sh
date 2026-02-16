#!/bin/bash
# AWS Deployment Script for TraderBot
# Usage: ./aws/deploy.sh [staging|production]

set -e

ENV=${1:-staging}
AWS_REGION=${AWS_REGION:-us-east-1}
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "Deploying TraderBot to $ENV in $AWS_REGION"

# Build and push Docker image
echo "Building Docker image..."
docker build -t traderbot:latest .

# Tag for ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Create ECR repo if not exists
aws ecr describe-repositories --repository-names traderbot || \
    aws ecr create-repository --repository-name traderbot

# Push image
docker tag traderbot:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/traderbot:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/traderbot:latest

# Register task definition
sed -e "s/\${AWS_ACCOUNT_ID}/$AWS_ACCOUNT_ID/g" \
    -e "s/\${AWS_REGION}/$AWS_REGION/g" \
    aws/ecs-task-definition.json > /tmp/task-def.json

aws ecs register-task-definition --cli-input-json file:///tmp/task-def.json

# Update ECS service
CLUSTER_NAME="traderbot-$ENV"
SERVICE_NAME="traderbot-api-$ENV"

aws ecs update-service \
    --cluster $CLUSTER_NAME \
    --service $SERVICE_NAME \
    --task-definition traderbot-api \
    --force-new-deployment

echo "Deployment initiated. Check ECS console for status."
