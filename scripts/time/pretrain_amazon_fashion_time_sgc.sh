cd "$(dirname $0)"

python3 ../../train_gnn.py \
-d amazon_fashion \
--bs 2048 \
--n_epoch 20 \
--lr 0.001 \
--n_runs 1 \
--gpu 3 \
--model sgc \
--n_hidden 256 \
--fanout 15,10,5 \
--data_type amazon \
--task_type time_trans \
--mode pretrain \
--seed 0