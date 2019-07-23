# train
~/anaconda3/bin/python train.py --gpu_ids 0,1 --net_idx 0 --no_flip --num_nets 8 --name cars.merged.context.0 --label_nc 0 --no_instance --loadSize 256 --input_nc 3 --output_nc 1 --batchSize 64 --display_freq 600 --dataroot /home/shared/datasets/cars.merged.new 

# test
--gpu_ids 0,1 --net_idx 0 --no_flip --num_nets 8 --name cars.merged.context.0 --label_nc 0 --no_instance --loadSize 256 --input_nc 3 --output_nc 1 --batchSize 1 --dataroot /home/shared/datasets/cars.merged.new --phase