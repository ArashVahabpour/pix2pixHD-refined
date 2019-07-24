# tonmoy server

# tab 1
cd ~/Desktop/pix2pixHD-refined/stage_2__kfold_and_refine/stage_2_1__kfold; ~/anaconda3/bin/python train.py --gpu_ids 0,1 --net_idx 0 --no_flip --num_nets 8 --name cars.merged.context.0 --label_nc 0 --loadSize 256 --input_nc 3 --output_nc 1 --batchSize 64 --display_freq 600 --dataroot /home/shared/datasets/cars.merged.new --checkpoints_dir /home/arash/Desktop/checkpoints
cd ~/Desktop/pix2pixHD-refined/stage_2__kfold_and_refine/stage_2_1__kfold; ~/anaconda3/bin/python train.py --gpu_ids 0,1 --net_idx 1 --no_flip --num_nets 8 --name cars.merged.context.1 --label_nc 0 --loadSize 256 --input_nc 3 --output_nc 1 --batchSize 64 --display_freq 600 --dataroot /home/shared/datasets/cars.merged.new --checkpoints_dir /home/arash/Desktop/checkpoints
cd ~/Desktop/pix2pixHD-refined/stage_2__kfold_and_refine/stage_2_1__kfold; ~/anaconda3/bin/python train.py --gpu_ids 0,1 --net_idx 2 --no_flip --num_nets 8 --name cars.merged.context.2 --label_nc 0 --loadSize 256 --input_nc 3 --output_nc 1 --batchSize 64 --display_freq 600 --dataroot /home/shared/datasets/cars.merged.new --checkpoints_dir /home/arash/Desktop/checkpoints


# tab 2
cd ~/Desktop/pix2pixHD-refined/stage_2__kfold_and_refine/stage_2_1__kfold; ~/anaconda3/bin/python train.py --gpu_ids 2,3 --net_idx 3 --no_flip --num_nets 8 --name cars.merged.context.3 --label_nc 0 --loadSize 256 --input_nc 3 --output_nc 1 --batchSize 24 --display_freq 600 --dataroot /home/shared/datasets/cars.merged.new --checkpoints_dir /home/arash/Desktop/checkpoints
cd ~/Desktop/pix2pixHD-refined/stage_2__kfold_and_refine/stage_2_1__kfold; ~/anaconda3/bin/python train.py --gpu_ids 2,3 --net_idx 5 --no_flip --num_nets 8 --name cars.merged.context.5 --label_nc 0 --loadSize 256 --input_nc 3 --output_nc 1 --batchSize 24 --display_freq 600 --dataroot /home/shared/datasets/cars.merged.new --checkpoints_dir /home/arash/Desktop/checkpoints


# tab 3
cd ~/Desktop/pix2pixHD-refined/stage_2__kfold_and_refine/stage_2_1__kfold; ~/anaconda3/bin/python train.py --gpu_ids 4,5 --net_idx 6 --no_flip --num_nets 8 --name cars.merged.context.6 --label_nc 0 --loadSize 256 --input_nc 3 --output_nc 1 --batchSize 24 --display_freq 600 --dataroot /home/shared/datasets/cars.merged.new --checkpoints_dir /home/arash/Desktop/checkpoints
cd ~/Desktop/pix2pixHD-refined/stage_2__kfold_and_refine/stage_2_1__kfold; ~/anaconda3/bin/python train.py --gpu_ids 4,5 --net_idx 7 --no_flip --num_nets 8 --name cars.merged.context.7 --label_nc 0 --loadSize 256 --input_nc 3 --output_nc 1 --batchSize 24 --display_freq 600 --dataroot /home/shared/datasets/cars.merged.new --checkpoints_dir /home/arash/Desktop/checkpoints




#~~~~~~~~~~~~~~~

# qiujing server

~/anaconda3/bin/python train.py --gpu_ids 0,1 --net_idx 4 --no_flip --num_nets 8 --name cars.merged.context.4 --label_nc 0 --loadSize 256 --input_nc 3 --output_nc 1 --batchSize 24 --display_freq 600 --dataroot /media/qiujing/91ec90ab-87ac-41f3-9f23-fbbbf9c36c61/arash/datasets/cars.merged.new --continue_train

