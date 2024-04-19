# A_VALUES=("coco" "db")  # 可能的字符串值
B_VALUES=("hidden" "mbrs" "cin" "pimog" "advmark" "adv" "signature")  # 可能的字符串值
C_VALUES=(5)  # 可能的整数值

for ((j=0; j<${#B_VALUES[@]}; j++)); do
  b=${B_VALUES[j]}
  python3 train.py --epochs 300 --load 1 --batch-size 250 --org-dir ../../coco/train/train_class/ --defense $b --wm-dir ../../coco/val_wm/$b/ --out-dir ./models/$b.pth >$b.txt
done