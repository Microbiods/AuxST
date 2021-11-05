#!/bin/bash
# alexnet, vgg19, resnet101, densenet121, inception_v3, resnext101_32x8d, efficientnet_b7

# model=alexnet
# python3 model_baseline.py --pred_root output/baseline_model_${model}/ --batch 32 --workers 8 --model ${model} 

# model=vgg19
# python3 model_baseline.py --pred_root output/baseline_model_${model}/ --batch 32 --workers 8 --model ${model}


# declare -a StringArray=("alexnet" "vgg19" "resnet101" "densenet121" "inception_v3" "resnext101_32x8d" "efficientnet_b7" )
# model 
# declare -a StringArray=("efficientnet_b4")

# for model in ${StringArray[@]}
# do python3 model_baseline.py --pred_root output/baseline_model/${model}/ --batch 32 --workers 8 --model ${model} 
# done

declare -a StringArray=("alexnet" "vgg19" "inception_v3" "resnet101" "densenet121" "efficientnet_b4" "ViT")
for model in ${StringArray[@]}
do
 
python3 baseline.py --model ${model}

fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh

done





# declare -a StringArray=("alexnet" "vgg19" "resnet101" "densenet121" "inception_v3" "resnext101_32x8d" "efficientnet_b4" )
# # declare -a StringArray=("efficientnet_b4" )
# # model 
# # declare -a StringArray=("efficientnet_b4")

# for seed in ${StringArray[@]}
# for model in ${StringArray[@]}
# do python3 model_baseline_v1.py --pred_root output/baseline_mode_v1/${model}/ --batch 32 --workers 8 --model ${model} 
# done


# declare -a StringArray=("vgg19" "inception_v3" "resnet101" "densenet121" "efficientnet_b4" "ViT")

# declare -a StringArray=("alexnet")
# for model in ${StringArray[@]}
# do 
# for ((i=0; i <= 9; i++))
# do python3 model_baseline_v3.py --pred_root output/baseline_model_v3/${model}/seed_${i}/ --batch 32 --workers 8 --model ${model} --seed ${i}
# done
# done




# # TL
# declare -a StringArray=("baseline" "fine_tuning")
# # declare -a StringArray=("baseline" "pretrain_3fcs" "fine_tuning" "random_init")

# for model in ${StringArray[@]}
# do python3 transfer_baseline.py --pred_root output/model_TL/${model}/ --batch 32 --workers 8 --model ${model} --debug
# done



# TL

# declare -a StringArray=("baseline" "pretrain_3fcs" "fine_tuning" "random_init" "train_allfcs")


# declare -a StringArray=("baseline")
# for model in ${StringArray[@]}
# # do
# # for ((i=0; i <= 3; i++))
# do python3 transfer_baseline_v1.py --pred_root output/model_TL_v1_test/${model}/seed_${i}/ --batch 32 --workers 8 --model ${model} --seed 0 
# # done
# done

# # Windows
# declare -a StringArray=( "128" "150" "224" "299" "512")
# # declare -a StringArray=("baseline" "pretrain_3fcs" "fine_tuning" "random_init")

# for win in ${StringArray[@]}
# do python3 window_baseline.py --pred_root output/window_baseline/win_${win}/ --batch 32 --workers 8 --window ${win} 
# done


# # Resolution
# declare -a StringArray=( "128" "224" "299" "384" "512")
# # declare -a StringArray=("baseline" "pretrain_3fcs" "fine_tuning" "random_init")

# for res in ${StringArray[@]}
# do python3 resolution_baseline.py --pred_root output/resolution_baseline/resolution_${res}/ --batch 32 --workers 8 --resolution ${res} 
# done



# model=densenet121
# python3 model_baseline.py --pred_root output/baseline_model_${model}/ --batch 32 --workers 8 --model ${model}

# model=inception_v3
# python3 model_baseline.py --pred_root output/baseline_model_${model}/ --batch 32 --workers 8 --model ${model}

# model=resnext101_32x8d
# python3 model_baseline.py --pred_root output/baseline_model_${model}/ --batch 32 --workers 8 --model ${model}

# model=efficientnet_b7
# python3 model_baseline.py --pred_root output/baseline_model_${model}/ --batch 16 --workers 8 --model ${model}


# nohup bash training.sh  > out.log 2>&1 &
