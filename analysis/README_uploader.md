# W&B Rank Dynamics Uploader

This tool automatically computes and uploads rank dynamics metrics to your existing W&B runs, enabling interactive analysis in the W&B interface.

## Features

✅ **All Rank Measures**: Supports `effective_rank`, `approximate_rank`, `l1_distribution_rank`, `numerical_rank`  
✅ **All Dynamics Metrics**: Computes `rank_drop_gini`, `rank_decay_centroid`, `normalized_aurc` for each rank measure  
✅ **Duplicate Prevention**: Safely handles re-runs without creating duplicate data  
✅ **Ongoing Runs**: Only processes new steps for running experiments  
✅ **Dry Run Mode**: Test before making changes  
✅ **Batch Processing**: Handles hundreds of runs efficiently  
✅ **Progress Tracking**: Clear feedback on what's being processed  

## Quick Start

### 1. Test with a Single Run
```bash
cd analysis
python test_uploader.py
```

This will:
- Test with 3 runs in dry-run mode
- Show you what would be processed
- Optionally run on 1 real run to verify

### 2. Process All Runs (Dry Run First)
```bash
python wandb_rank_dynamics_uploader.py \
    --entity YOUR_ENTITY \
    --project YOUR_PROJECT \
    --dry-run
```

### 3. Process All Runs (For Real)
```bash
python wandb_rank_dynamics_uploader.py \
    --entity YOUR_ENTITY \
    --project YOUR_PROJECT
```

## Advanced Usage

### Specific Rank Types Only
```bash
python wandb_rank_dynamics_uploader.py \
    --entity YOUR_ENTITY \
    --project YOUR_PROJECT \
    --rank-types effective_rank approximate_rank
```

### Ratio Mode (Instead of Difference)
```bash
python wandb_rank_dynamics_uploader.py \
    --entity YOUR_ENTITY \
    --project YOUR_PROJECT \
    --analysis-mode ratio
```

### Limited Testing
```bash
python wandb_rank_dynamics_uploader.py \
    --entity YOUR_ENTITY \
    --project YOUR_PROJECT \
    --max-runs 10 \
    --dry-run
```

## What Gets Added to W&B

For each rank measure (e.g., `effective_rank`), you get three new metrics:

1. **`effective_rank_rank_drop_gini`** - Inequality in rank drops across layers
2. **`effective_rank_rank_decay_centroid`** - Which layers experience most decay  
3. **`effective_rank_normalized_aurc`** - Overall rank preservation

Multiply by 4 rank measures = **12 new metrics per run**!

## Using the Results in W&B

After running the uploader, go to your W&B project and:

### 1. Create Custom Charts
- Go to your project workspace
- Add new panel → Line plot
- X-axis: `_step`
- Y-axis: `effective_rank_rank_drop_gini` (or any other new metric)
- Group by: `config.net.type` or other config parameters

### 2. Compare Runs Interactively
- Use W&B's filtering: click "Add filter" 
- Filter by config parameters (e.g., `config.learner.step_size = 0.01`)
- View only runs with specific characteristics
- Compare their rank dynamics patterns

### 3. Create Reports
- Use W&B Reports to create analysis dashboards
- Include multiple rank dynamics charts
- Add markdown explanations of patterns
- Share with collaborators

## How It Works

### Duplicate Prevention
The uploader is smart about duplicates:

- **Finished runs**: Skips runs that already have rank dynamics
- **Ongoing runs**: Only processes new steps since last analysis
- **Safe re-running**: You can run the script multiple times safely

### Layer Ordering
The uploader automatically determines the correct layer order using:

1. **Model Reconstruction**: Reconstructs your model from W&B config to get exact layer order
2. **Indexed Fallback**: For indexed layer names (`layer_0`, `layer_1`, etc.)
3. **Manual Override**: For semantic names, requires manual specification

### Error Handling
- Graceful handling of missing data
- Clear error messages for problematic runs
- Continues processing even if some runs fail
- Detailed progress reporting

## Example Output

```
🚀 Starting rank dynamics analysis...
   Entity: your-entity
   Project: your-project
   Rank types: ['effective_rank', 'approximate_rank', 'l1_distribution_rank', 'numerical_rank']
   Analysis mode: difference

Found 156 runs in your-entity/your-project

Summary:
  Runs that will be processed: 89
  Runs that will be skipped: 67

Processing runs: 100%|████████| 156/156 [15:23<00:00,  1.2runs/s]

📊 Processing Complete!
  Total runs: 156
  Processed: 89
  Skipped: 67
  Errors: 0
  New steps added: 12,847
```

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the correct directory:
```bash
cd /path/to/architectural_and_learning_on_loss_landscape/analysis
python wandb_rank_dynamics_uploader.py --help
```

### Layer Order Issues
If the script can't determine layer order:
1. Check that your runs have the expected rank metrics
2. Look at the layer names printed in the error messages
3. Modify the `get_correct_layer_order()` function for your specific naming scheme

### W&B Authentication
Make sure you're logged into W&B:
```bash
wandb login
```

### Memory Issues
For very large projects, use `--max-runs` to process in batches:
```bash
# Process first 50 runs
python wandb_rank_dynamics_uploader.py --entity YOUR_ENTITY --project YOUR_PROJECT --max-runs 50

# Then next 50, etc.
```

## Integration with Existing Analysis

The uploader works seamlessly with your existing analysis scripts:

- **Notebooks**: Use the enhanced W&B interface for interactive exploration
- **Scripts**: Continue using `plot_rank_dynamics.py` for detailed individual run analysis
- **Reports**: Create W&B reports combining both approaches

## Safety Features

- **Dry run mode**: Always test first
- **Confirmation prompts**: Interactive confirmation before processing many runs
- **Error recovery**: Continues processing even if individual runs fail
- **Duplicate detection**: Won't overwrite existing data
- **Progress tracking**: Clear feedback on what's happening
