python train_stereo.py --batch_size 8 \
                       --train_iters 22 \
                       --valid_iters 32 \
                       --spatial_scale -0.2 0.4 \
                       --saturation_range 0 1.4 \
                       --n_downsample \
                       --num_steps 200000 \
                       --mixed_precision