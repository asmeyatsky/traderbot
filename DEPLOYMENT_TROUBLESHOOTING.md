# Cloud Run Deployment - Secret Configuration Guide

Based on the Terraform outputs, here are the correct values for GitHub secrets:

## GCP_SA_KEY
- This should contain the full service account key JSON
- The service account needs permissions: roles/run.admin, roles/storage.admin, roles/artifactregistry.writer

## DATABASE_URL
- Format: postgresql://trading:{password}@10.205.0.3:5432/traderbot
- The password needs to match what was used in the Terraform database setup
- The database name is "traderbot"
- The private IP address is 10.205.0.3

## REDIS_URL
- Format: redis://10.17.212.251:6379
- From Terraform output: redis_host = "10.17.212.251", redis_port = 6379

## Troubleshooting
1. Check GitHub Actions logs for specific error messages
2. Verify the service account in GCP_SA_KEY has proper permissions
3. Verify the database password in DATABASE_URL matches the Terraform configuration
4. Ensure the VPC connector is in "READY" state before deployment

## Root Cause Identified
The Docker build was failing due to native dependencies of spaCy and other ML packages.
Fixed by adding build-essential and cmake to system dependencies and upgrading pip.