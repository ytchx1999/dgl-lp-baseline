cd "$(dirname $0)"

python3 ../../train_gnn.py \
-d gowalla_Nightlife \
--bs 512 \
--n_epoch 10 \
--lr 0.001 \
--n_runs 1 \
--gpu 2 \
--model dgi \
--n_hidden 256 \
--fanout 15,10,5 \
--data_type gowalla \
--task_type time_trans \
--mode pretrain \
--seed 0