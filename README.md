# Complex-valued-few-shot-learning
The implement of “Few-Shot Learning with Complex-valued Neural Networks” BMVC2020


We will upload our code as soon as possible
'

The code use doc.
for mini 5 way-1shot exp
python train --gpu=0 --n_way=5 --n_shot=1--lr=0.001 --step_size=25000 --dataset=mini --exp_name=mini_complex_beta300_25000 --beta 300 --rn=300 --alpha=0.99 --k=20 --complex True
for mini 5 way-5shot exp
python train --gpu=0 --n_way=5 --n_shot=5--lr=0.001 --step_size=25000 --dataset=mini --exp_name=mini_complex_beta300_25000 --beta 300 --rn=300 --alpha=0.99 --k=20 --complex True 
for tiered 5 way-1shot exp
python train --gpu=0 --n_way=5 --n_shot=1--lr=0.001 --step_size=25000 --dataset=tiered --exp_name=tiered_complex_beta300_25000 --beta 300 --rn=300 --alpha=0.99 --k=20 --complex True 
for tiered 5 way-1shot exp
python train --gpu=0 --n_way=5 --n_shot=5--lr=0.001 --step_size=25000 --dataset=tiered --exp_name=tiered_complex_beta300_25000 --beta 300 --rn=300 --alpha=0.99 --k=20 --complex True 

There are other super-para for ablation exp.

