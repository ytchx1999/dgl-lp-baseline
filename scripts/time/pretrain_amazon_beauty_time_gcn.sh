cd "$(dirname $0)"

python3 ../../train_gnn.py \
-d amazon_beauty \
--bs 512 \
--n_epoch 20 \
--lr 0.001 \
--n_runs 1 \
--gpu 1 \
--model gcn \
--n_hidden 256 \
--fanout 15,10 \
--data_type amazon \
--task_type time_trans \
--mode pretrain \
--seed 0