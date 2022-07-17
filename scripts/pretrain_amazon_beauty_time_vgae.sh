cd "$(dirname $0)"

python3 ../train_vgae.py \
-d amazon_beauty \
--bs 512 \
--n_epoch 10 \
--lr 0.001 \
--n_runs 1 \
--gpu 0 \
--data_type amazon \
--task_type time_trans \
--mode pretrain 