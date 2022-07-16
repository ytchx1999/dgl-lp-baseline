# dgl-lp-baseline
Link prediction gnn baselines implemented by DGL.

## Environment

```bash
dgl == 0.8.2
torch == 1.8.2
sklearn == 1.0.2
tqdm
```

## Data

```bash
.
|-- processed_data
   |-- amazon
   |--|-- field_trans
   |--|--|-- ...
   |--|-- time_trans
   |--|--|-- ...
   |-- gowalla
   |--|-- field_trans
   |--|--|-- ...
   |--|-- time_trans
   |--|--|-- ml_gowalla_Entertainment_pretrain.csv
   |--|--|-- ml_gowalla_Entertainment_pretrain_node.npy
   |--|--|-- ml_gowalla_Entertainment_downstream.csv
   |--|--|-- ml_gowalla_Entertainment_downstream_node.npy
   |--|--|-- ...
```

## Run

```bash
cd scripts/
# pretrain + gowalla_Entertainment + time transfer: model GraphSAGE
bash pretrain_gowalla_env_time_sage.sh
# downstream + gowalla_Entertainment + time transfer: model GraphSAGE
bash downstream_gowalla_env_time_sage.sh
```
