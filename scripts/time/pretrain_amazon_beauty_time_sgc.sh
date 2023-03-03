cd "$(dirname $0)"

python3 ../../train_gnn.py \
-d amazon_beauty \
--bs 2048 \
--n_epoch 20 \
--lr 0.001 \
--n_runs 1 \
--gpu 2 \
--model sgc \
--n_hidden 256 \
--k_hop 3 \
--data_type amazon \
--task_type time_trans \
--mode pretrain \
--seed 0