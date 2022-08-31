cd "$(dirname $0)"

python3 ../../train_gnn.py \
-d amazon_fashion \
--bs 512 \
--n_epoch 20 \
--lr 0.001 \
--n_runs 3 \
--gpu 3 \
--model gat \
--n_hidden 256 \
--n_heads 2 \
--fanout 15,10,5 \
--data_type amazon \
--task_type tf_trans \
--mode pretrain \
--seed 0