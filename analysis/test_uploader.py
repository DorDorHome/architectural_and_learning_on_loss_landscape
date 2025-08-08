#!/usr/bin/env python3
"""
Test script for the W&B rank dynamics uploader.
Tests the functionality with a single run first.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.wandb_rank_dynamics_uploader import add_rank_dynamics_to_existing_runs

def test_single_run():
    """Test with a single run first."""
    print("🧪 Testing W&B Rank Dynamics Uploader")
    print("=" * 50)
    
    # Get user input
    entity = input("Enter your W&B entity: ")
    project = input("Enter your W&B project: ")
    
    print(f"\n🔍 Testing with entity: {entity}, project: {project}")
    print("📊 Running in DRY RUN mode (max 3 runs)")
    
    # Test with dry run first
    stats = add_rank_dynamics_to_existing_runs(
        entity=entity,
        project=project,
        rank_types=['effective_rank'],  # Test with just one rank type first
        analysis_mode='difference',
        dry_run=True,
        max_runs=3  # Limit to 3 runs for testing
    )
    
    print(f"\n📊 Test Results:")
    print(f"  Total runs: {stats['total_runs']}")
    print(f"  Would process: {stats['processed_runs']}")
    print(f"  Would skip: {stats['skipped_runs']}")
    print(f"  Errors: {stats['error_runs']}")
    print(f"  New steps would be added: {stats['new_steps_added']}")
    
    if stats['processed_runs'] > 0:
        print(f"\n✅ Test successful! Ready to run with actual logging.")
        
        confirm = input(f"\nRun for real on ONE run? (y/N): ")
        if confirm.lower() == 'y':
            print(f"🚀 Running with actual W&B logging (1 run only)...")
            stats_real = add_rank_dynamics_to_existing_runs(
                entity=entity,
                project=project,
                rank_types=['effective_rank'],
                analysis_mode='difference',
                dry_run=False,
                max_runs=1
            )
            
            print(f"\n📊 Real Run Results:")
            print(f"  Processed: {stats_real['processed_runs']}")
            print(f"  New steps added: {stats_real['new_steps_added']}")
            print(f"\n✅ Check your W&B project to see the new metrics!")
        else:
            print("Test complete. No actual logging performed.")
    else:
        print(f"\n⚠️  No runs would be processed. Check your entity/project names.")

if __name__ == "__main__":
    test_single_run()
