A_VALUES=("coco" "db")  # 可能的字符串值
B_VALUES=("mbrs" "DwtDctSvd" "hidden" "adv" "signature" "pimog" "mbrs" "stega")  # 可能的字符串值
C_VALUES=(5)  # 可能的整数值

for ((i=0; i<${#A_VALUES[@]}; i++)); do
  for ((j=0; j<${#B_VALUES[@]}; j++)); do
    a=${A_VALUES[i]}
    b=${B_VALUES[j]}
    cmd=
    echo "python3 finetune.py --mode eval --batch 10 --lamda_i 5 --lr_image 5e-4 --iter_finetune 250 --defense $b --dataset-folder ./dataset/$a > ./result/$a/$b.txt"
    python3 finetune.py --mode train --batch 1 --lamda_i 5 --lr_image 5e-4 --iter_E 1 --iter_finetune 250 --defense $b --dataset-folder ./dataset/$a > ./temp.txt
  done
done