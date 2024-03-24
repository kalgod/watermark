VALUES=("jpeg" "regen-diff" "regen-bmsh" "regen-cheng" )  # 可能的字符串值
C_VALUES=(5 6 7 8 9 10 11 12 13 14 15)  # 可能的整数值

# for a in "${A_VALUES[@]}"; do
#   for b in "${B_VALUES[@]}"; do
#     for c in "${C_VALUES[@]}"; do
#       python3 finetune.py --mode eval --checkpoint /home/cjh/work/watermark/WEvade/finetuned/epoch_100_0.0005_0.0005_10_0_wevade_onlywevade.pth --dataset-folder ./dataset/coco/val --attack_train $a --attack_train1 $b --lamda_i $c --iter_finetune 300  > "${a}_${b}_${c}.txt"
#     done
#   done
# done

for ((i=0; i<${#VALUES[@]}; i++)); do
  for ((j=i+1; j<${#VALUES[@]}; j++)); do
    a=${VALUES[i]}
    b=${VALUES[j]}
    for c in "${C_VALUES[@]}"; do
      python3 finetune.py --mode eval --tau 0.7 --batch 10 --checkpoint ./finetuned/epoch_100_0.0005_0.0005_10_0_wevade_onlywevade.pth --dataset-folder ./dataset/coco/val --attack_train $a --attack_train1 $b --lamda_i $c --iter_finetune 300  > "${a}_${b}_${c}.txt"
    done
  done
done