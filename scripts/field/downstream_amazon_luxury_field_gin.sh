cd "$(dirname $0)"

python3 ../../train_gnn.py \
-d amazon_luxury \
--bs 512 \
--n_epoch 20 \
--lr 0.001 \
--n_runs 3 \
--gpu 3 \
--model gin \
--n_hidden 256 \
--fanout 15,10,5 \
--data_type amazon \
--task_type field_trans \
--mode downstream \
--seed 0