#!/usr/bin/env python3
"""
Test script to verify data can be fetched from icechunk repository
"""

import os
import numpy as np
import zarr
import icechunk

# Set AWS credentials
os.environ['AWS_ACCESS_KEY_ID'] = 'your_aws_access_key_id'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_aws_secret_access_key'
os.environ['AWS_REGION'] = 'us-east-1'

def test_icechunk_data_fetch():
    """Test fetching data from the icechunk repository."""
    
    print("🔍 Testing IceChunk Data Retrieval")
    print("=" * 50)
    
    try:
        # Find the most recent repository
        import boto3
        s3 = boto3.client('s3')
        paginator = s3.get_paginator('list_objects_v2')
        
        repositories = []
        for page in paginator.paginate(Bucket='paramlake', Prefix='mnist-'):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if 'icechunk' in key or key.endswith('.json'):
                        repo_prefix = key.split('/')[0]
                        if repo_prefix not in repositories:
                            repositories.append(repo_prefix)
        
        if not repositories:
            print("❌ No repositories found")
            return
        
        # Use the most recent repository (last in alphabetical order usually means most recent)
        repo_prefix = sorted(repositories)[-1]
        print(f"📁 Using repository: {repo_prefix}")
        
        # Connect to the repository
        storage = icechunk.s3_storage(
            bucket="paramlake", 
            prefix=repo_prefix,
            from_env=True
        )
        
        repo = icechunk.Repository.open(storage)
        print(f"✅ Successfully opened repository")
        
        # List branches
        branches = list(repo.list_branches())
        print(f"📂 Available branches: {branches}")
        
        # Check data on each branch
        for branch in ['adam', 'sgd']:
            if branch in branches:
                print(f"\n🔍 Checking branch: {branch}")
                
                try:
                    session = repo.readonly_session(branch)
                    store = session.store
                    
                    # Open as zarr group
                    group = zarr.open_group(store, mode="r")
                    print(f"  📊 Root groups: {list(group.keys())}")
                    
                    # Check parameters data (the actual structure used)
                    if 'parameters' in group:
                        params_group = group['parameters']
                        print(f"  📋 Parameters group structure: {list(params_group.keys())}")
                        
                        # Recursively explore the parameters structure
                        for key in params_group.keys():
                            item = params_group[key]
                            print(f"  📂 '{key}' type: {type(item)}")
                            
                            if hasattr(item, 'keys'):  # It's a group
                                sub_keys = list(item.keys())
                                print(f"    📋 Contains: {sub_keys}")
                                
                                # If it contains arrays, show their info
                                for sub_key in sub_keys:
                                    sub_item = item[sub_key]
                                    if hasattr(sub_item, 'shape'):  # It's an array
                                        print(f"      📊 '{sub_key}' shape: {sub_item.shape}, dtype: {sub_item.dtype}")
                                        
                                        # Try to read a small sample
                                        try:
                                            sample_data = sub_item[:]
                                            print(f"      📈 Stats: min={np.min(sample_data):.6f}, max={np.max(sample_data):.6f}, mean={np.mean(sample_data):.6f}")
                                            print(f"      ✅ Successfully read parameter data!")
                                        except Exception as e:
                                            print(f"      ⚠️ Could not read data: {e}")
                                    elif hasattr(sub_item, 'keys'):  # Nested group
                                        nested_keys = list(sub_item.keys())
                                        print(f"      📂 '{sub_key}' contains: {nested_keys}")
                            elif hasattr(item, 'shape'):  # It's an array directly
                                print(f"    📊 Direct array shape: {item.shape}, dtype: {item.dtype}")
                                try:
                                    sample_data = item[:]
                                    print(f"    📈 Stats: min={np.min(sample_data):.6f}, max={np.max(sample_data):.6f}, mean={np.mean(sample_data):.6f}")
                                    print(f"    ✅ Successfully read parameter data!")
                                except Exception as e:
                                    print(f"    ⚠️ Could not read data: {e}")
                    else:
                        print(f"  ⚠️  No 'parameters' group found")
                        
                except Exception as e:
                    print(f"  ❌ Error accessing branch {branch}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Get commit history
        print(f"\n📝 Recent commit history:")
        try:
            session = repo.readonly_session("main")
            snapshot_id = session.snapshot_id
            
            history = repo.ancestry(snapshot_id=snapshot_id)
            count = 0
            for ancestor in history:
                if count < 5:  # Show last 5 commits
                    print(f"  🆔 {ancestor.id}")
                    print(f"     📅 {ancestor.written_at}")
                    print(f"     💬 {ancestor.message}")
                    count += 1
                else:
                    break
        except Exception as e:
            print(f"  ❌ Error getting commit history: {e}")
            
        print(f"\n✅ Data retrieval test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in data retrieval test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_icechunk_data_fetch() 