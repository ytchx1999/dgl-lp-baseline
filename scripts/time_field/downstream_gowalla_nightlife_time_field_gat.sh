cd "$(dirname $0)"

python3 ../../train_gnn.py \
-d gowalla_Nightlife \
--bs 512 \
--n_epoch 20 \
--lr 0.001 \
--n_runs 3 \
--gpu 0 \
--model gat \
--n_hidden 256 \
--n_heads 2 \
--fanout 15,10,5 \
--data_type gowalla \
--task_type tf_trans \
--mode pretrain \
--seed 0