for imb_factor in 0.1
do
for corruption_type in 'unif'
do
for corruption_prob in 0.6
do
CUDA_VISIBLE_DEVICES=0 python3 ./experiment/train_cifar_ssl.py --dataset cifar10 --num_class 10 --warm_up 30 --imb_factor ${imb_factor} --corruption_type ${corruption_type} --corruption_prob ${corruption_prob} --warmup_SBCL --train_SBCL --is_prototypical_selection --proto_assign_type 'DaCC' --train_MIDL --exp-str CIFAR10_DaSC
done
done
done
