#!/usr/bin/env bash
# test

cd $0 
cd ..

for NET_IDX in $(seq 0 7)
do
	~/anaconda3/bin/python test.py --gpu_ids 0 --net_idx $NET_IDX --no_flip --num_nets 8 --name cars.merged.context.$NET_IDX --label_nc 0 --no_instance --loadSize 256 --input_nc 3 --output_nc 1 --batchSize 1 --dataroot /home/shared/datasets/cars.merged.new --phase val
done
