#!/bin/bash

# declare -a StringArray=("alexnet" "vgg19" "inception_v3" "resnet101" "densenet121" "efficientnet_b4" "ViT")
# for model in ${StringArray[@]}
# do
 
# python3 baseline.py --model ${model} --pred_root output/model/${model}/

# fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh

# done

### return efficientnet_b4



# declare -a StringArray=("ftfc" "ftconv" "ftall" ) # all are not freezed
# for ft in ${StringArray[@]}
# do

# python3 baseline.py --model efficientnet_b4 --pretrained --finetuning ${ft} --pred_root output/transfer_pretrained/${ft}/

# fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh

# done

# return random init

# 




# ###  cropped window sizes 128 150 299 384 512 and all resize to 224

# declare -a StringArray=( "128" "150" "299" "384" "512" )
# for win in ${StringArray[@]}
# do

# python3 baseline.py --model efficientnet_b4 --finetuning ftall --window ${win} --pred_root output/window_init/${win}/ --debug

# fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh

# done

# ### epoch = 30 may be enough   

# # return 299





# #### 11.25, comment the resize before run this

# ## resolution to resize, default resolution is 224, window size 299


# ### 299 is unless
# # declare -a StringArray=( "128" "299" "384" "512" )

# declare -a StringArray=( "299" "384" "512" )
# for res in ${StringArray[@]}
# do

# python3 baseline.py --model efficientnet_b4 --finetuning ftall --window 299 --resolution ${res} --pred_root output/resolution/${res}/ --batch 8 --debug 

# fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh

# done

## and resolutions



# ### use aux loss (weight for aux from 0 to 1, gene ratio from 250 (10%) to (100%) )
# ### aux_ratio=0.1 (), aux_weight=1

# declare -a StringArray=( "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0")
# for i in ${StringArray[@]}
# do

# python3 baseline.py --model efficientnet_b4 --finetuning ftall --window 299 --resolution 224 --aux_ratio ${i} --pred_root output/auxnet/ratio_${i} --batch 32 

# fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh

# done






### use aux loss (weight for aux from 0 to 1, gene ratio from 250 (10%) to (100%) ) ### lost
### aux_weight=i (), aux_ratio = 1

# declare -a StringArray=("0.0001" "0.001" "0.01" "0.05" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "10") # "1.0 means the same with previous
# for i in ${StringArray[@]}
# do

# python3 baseline.py --model efficientnet_b4 --finetuning ftall --window 299 --resolution 224 --aux_weight ${i} --pred_root output/auxnet/weight_${i} --batch 32 

# fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh

# done


# declare -a StringArray=("100" "1000" "10000" "100000" "1000000") # "1.0 means the same with previous
# for i in ${StringArray[@]}
# do

# python3 baseline.py --model efficientnet_b4 --finetuning ftall --window 299 --resolution 224 --aux_weight ${i} --pred_root output/auxnet/weight_${i} --batch 32 

# fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh

# done



# declare -a StringArray=("2" "3" "4" "5" "6" "7" "8" "9") # "1.0 means the same with previous
# for i in ${StringArray[@]}
# do

# python3 baseline.py --model efficientnet_b4 --finetuning ftall --window 299 --resolution 224 --aux_weight ${i} --pred_root output/auxnet/weight_${i} --batch 32 

# fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh

# done




declare -a StringArray=("20" "30" "40" "50" "60" "70" "80" "90") # "1.0 means the same with previous
for i in ${StringArray[@]}
do

python3 baseline.py --model efficientnet_b4 --finetuning ftall --window 299 --resolution 224 --aux_weight ${i} --pred_root output/auxnet/weight_${i} --batch 32 

fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh

done


### use lstm (weight for aux from 0 to 1, gene ratio from 250 (10%) to (100%) )


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


# nohup bash training.sh  > output/logging/aux_weight.log 2>&1 &
python3 baseline.py --model efficientnet_b4 --finetuning ftall --window 299 --resolution 224 --aux_ratio 1 --aux_weight 1 --pred_root output/auxnet/weight_${i} --batch 32 --debug