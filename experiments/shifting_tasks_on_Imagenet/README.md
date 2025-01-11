# This repo produces a more robusts, more flexible implementation for training on shifting tasks, specifically on Imagenet


## apparent mistake (deliberate?) in the original implementation:

in new_head setting, weights are set to zero in the original implementation by the authors. In our implementation, the new weights in the new head is reinitialized using kaiming init.