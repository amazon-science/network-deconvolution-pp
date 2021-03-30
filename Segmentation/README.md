# Semantic Segmentation

### Training from scratch on the Cityscapes dataset 

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cityscapes --model deeplabv3_resnet50 -b 8 --epochs 30 --deconv False --pretrained-backbone False --lr 0.1 --base-size 512 &  

CUDA_VISIBLE_DEVICES=1 python train.py --dataset cityscapes --model deeplabv3_resnet50d -b 8 --epochs 30  --deconv True --pretrained-backbone False --lr 0.1 --base-size 512 &

#### Training with distributed data parallel 

python -u -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --dataset cityscapes --model deeplabv3_resnet50 -b 2 --epochs 200  --pretrained-backbone False --lr 0.1  --base-size 1024 

python -u -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --dataset cityscapes --model deeplabv3_resnet50d -b 2 --epochs 200  --deconv True --pretrained-backbone False --lr 0.1  --sync True --base-size 1024 
