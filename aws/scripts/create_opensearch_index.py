#!/usr/bin/env python3
"""
Create OpenSearch Serverless vector index for Bedrock Knowledge Base.

This script creates the required index with proper mappings for
Amazon Bedrock Knowledge Base vector search.
"""

import argparse
import boto3
import json
import time
import sys
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

def get_collection_endpoint(collection_name: str, region: str) -> str:
    """Get the collection endpoint from AWS."""
    client = boto3.client('opensearchserverless', region_name=region)
    
    # List collections
    response = client.list_collections(
        collectionFilters={
            'name': collection_name
        }
    )
    
    if not response.get('collectionSummaries'):
        raise ValueError(f"Collection '{collection_name}' not found")
    
    collection_id = response['collectionSummaries'][0]['id']
    
    # Get collection details
    response = client.batch_get_collection(ids=[collection_id])
    
    if not response.get('collectionDetails'):
        raise ValueError(f"Could not get details for collection '{collection_name}'")
    
    endpoint = response['collectionDetails'][0]['collectionEndpoint']
    return endpoint


def create_index(
    endpoint: str,
    index_name: str,
    region: str,
    vector_field: str = "embedding",
    text_field: str = "text",
    metadata_field: str = "metadata",
    dimension: int = 1024,
    max_retries: int = 10,
    retry_delay: int = 30
):
    """Create the vector index in OpenSearch Serverless."""
    
    # Get AWS credentials
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        'aoss',
        session_token=credentials.token
    )
    
    # Clean endpoint (remove https:// if present)
    host = endpoint.replace('https://', '')
    
    # Create client
    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60
    )
    
    # Index mapping for Bedrock Knowledge Base
    index_body = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 512
            }
        },
        "mappings": {
            "properties": {
                vector_field: {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss",
                        "parameters": {
                            "m": 16,
                            "ef_construction": 512
                        }
                    }
                },
                text_field: {
                    "type": "text",
                    "index": True
                },
                metadata_field: {
                    "type": "text",
                    "index": False
                }
            }
        }
    }
    
    # Try to create index with retries
    for attempt in range(1, max_retries + 1):
        try:
            # Check if index exists
            if client.indices.exists(index=index_name):
                print(f"Index '{index_name}' already exists")
                return True
            
            # Create index
            print(f"Creating index '{index_name}' (attempt {attempt}/{max_retries})...")
            response = client.indices.create(index=index_name, body=index_body)
            
            if response.get('acknowledged'):
                print(f"Index '{index_name}' created successfully!")
                return True
            else:
                print(f"Unexpected response: {response}")
                
        except Exception as e:
            error_msg = str(e)
            
            if "403" in error_msg or "Forbidden" in error_msg:
                print(f"Access denied (attempt {attempt}/{max_retries}). "
                      f"Data access policy may still be propagating...")
                if attempt < max_retries:
                    print(f"Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Policy may need more time to propagate.")
                    print("You can try again in a few minutes.")
                    return False
            else:
                print(f"Error: {e}")
                return False
    
    return False


def main():
    parser = argparse.ArgumentParser(description='Create OpenSearch Serverless vector index')
    parser.add_argument('--environment', '-e', default='dev', help='Environment (dev, staging, prod)')
    parser.add_argument('--project', '-p', default='trading-bot', help='Project name')
    parser.add_argument('--region', '-r', default='us-east-1', help='AWS region')
    parser.add_argument('--dimension', '-d', type=int, default=1024, help='Vector dimension')
    parser.add_argument('--max-retries', type=int, default=10, help='Maximum retry attempts')
    parser.add_argument('--retry-delay', type=int, default=30, help='Seconds between retries')
    
    args = parser.parse_args()
    
    collection_name = f"{args.project}-{args.environment}-vectors"
    index_name = f"{args.project}-trading-index"
    
    print(f"Configuration:")
    print(f"  Collection: {collection_name}")
    print(f"  Index:      {index_name}")
    print(f"  Region:     {args.region}")
    print(f"  Dimension:  {args.dimension}")
    print()
    
    try:
        # Get collection endpoint
        print("Getting collection endpoint...")
        endpoint = get_collection_endpoint(collection_name, args.region)
        print(f"  Endpoint: {endpoint}")
        print()
        
        # Create index
        success = create_index(
            endpoint=endpoint,
            index_name=index_name,
            region=args.region,
            dimension=args.dimension,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay
        )
        
        if success:
            print("\n✅ Index creation complete!")
            sys.exit(0)
        else:
            print("\n❌ Index creation failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
