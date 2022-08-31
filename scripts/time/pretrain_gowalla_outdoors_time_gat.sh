cd "$(dirname $0)"

python3 ../../train_gnn.py \
-d gowalla_Outdoors \
--bs 512 \
--n_epoch 10 \
--lr 0.001 \
--n_runs 1 \
--gpu 3 \
--model gat \
--n_hidden 256 \
--n_heads 2 \
--fanout 15,10,5 \
--data_type gowalla \
--task_type time_trans \
--mode pretrain \
--seed 0