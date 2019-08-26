######## Multi-GPU training example #######
~/anaconda3/bin/python train.py --name label2city_512p --batchSize 32 --gpu_ids 2,3,4,5 --checkpoints_dir /home/arash/Desktop/checkpoints__stage_4/ --dataroot /home/shared/datasets/cityscapes.pix2pixHD.folders/ --no_instance
