VALUES=("jpeg" "combined")  # 可能的字符串值
C_VALUES=(5)  # 可能的整数值
defense="adv"

for ((i=0; i<${#VALUES[@]}; i++)); do
  for ((j=i+1; j<${#VALUES[@]}; j++)); do
    a=${VALUES[i]}
    b=${VALUES[j]}
    for c in "${C_VALUES[@]}"; do
      echo "python3 finetune.py --mode eval --batch 4 --tau 0.8 --lr_image 5e-4 --defense $defense --dataset-folder ./dataset/coco --checkpoint ./finetuned/epoch_100_0.0005_0.0005_10_0_wevade_onlywevade.pth --attack_train $a --attack_train1 $b --lamda_i $c --iter_finetune 2000 --iteration 100 > "./result/coco/${defense}.txt""
      python3 finetune.py --mode eval --batch 10 --tau 0.8 --lr_image 5e-4 --defense $defense --dataset-folder ./dataset/db --checkpoint ./ckpt/coco_adv_train.pth --attack_train $a --attack_train1 $b --lamda_i $c --iter_finetune 2000 --iteration 100 > "./result/db/${defense}/${defense}.txt"
    done
  done
done