cd "$(dirname $0)"

python3 ../train_gnn.py \
-d gowalla_Entertainment \
--bs 512 \
--n_epoch 20 \
--lr 0.001 \
--n_runs 1 \
--gpu 1 \
--data_type gowalla \
--task_type time_trans \
--mode downstream 