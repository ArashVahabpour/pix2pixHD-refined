#!/usr/bin/env bash
# test
{--gpu_ids 0,1 --net_idx 0 --no_flip --num_nets 8 --name cars.merged.context.0 --label_nc 0 --no_instance --loadSize 256 --input_nc 3 --output_nc 1 --batchSize 1 --dataroot /home/shared/datasets/cars.merged.new --phase val
--gpu_ids 2,3 --net_idx 1 --no_flip --num_nets 8 --name cars.merged.context.1 --label_nc 0 --no_instance --loadSize 256 --input_nc 3 --output_nc 1 --batchSize 1 --dataroot /home/shared/datasets/cars.merged.new --phase val
--gpu_ids 4,5 --net_idx 2 --no_flip --num_nets 8 --name cars.merged.context.2 --label_nc 0 --no_instance --loadSize 256 --input_nc 3 --output_nc 1 --batchSize 1 --dataroot /home/shared/datasets/cars.merged.new --phase val} &

{--gpu_ids 0,1 --net_idx 3 --no_flip --num_nets 8 --name cars.merged.context.3 --label_nc 0 --no_instance --loadSize 256 --input_nc 3 --output_nc 1 --batchSize 1 --dataroot /home/shared/datasets/cars.merged.new --phase val
--gpu_ids 2,3 --net_idx 4 --no_flip --num_nets 8 --name cars.merged.context.4 --label_nc 0 --no_instance --loadSize 256 --input_nc 3 --output_nc 1 --batchSize 1 --dataroot /home/shared/datasets/cars.merged.new --phase val
--gpu_ids 4,5 --net_idx 5 --no_flip --num_nets 8 --name cars.merged.context.2 --label_nc 0 --no_instance --loadSize 256 --input_nc 3 --output_nc 1 --batchSize 1 --dataroot /home/shared/datasets/cars.merged.new --phase val} &

{--gpu_ids 0,1 --net_idx 6 --no_flip --num_nets 8 --name cars.merged.context.5 --label_nc 0 --no_instance --loadSize 256 --input_nc 3 --output_nc 1 --batchSize 1 --dataroot /home/shared/datasets/cars.merged.new --phase val
--gpu_ids 2,3 --net_idx 7 --no_flip --num_nets 8 --name cars.merged.context.6 --label_nc 0 --no_instance --loadSize 256 --input_nc 3 --output_nc 1 --batchSize 1 --dataroot /home/shared/datasets/cars.merged.new --phase val} &
