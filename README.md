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

Take **GraphSAGE** as an example, other methods (SGC, GIN, GAT) have the same operations and just replace `sage` in scripts name. 

### Time transfer

```bash
cd scripts/time/
# amazon
# amazon_beauty
bash pretrain_amazon_beauty_time_sage.sh
bash downstream_amazon_beauty_time_sage.sh
# amazon_fashion
bash pretrain_amazon_fashion_time_sage.sh
bash downstream_amazon_fashion_time_sage.sh
# amazon_luxury
bash pretrain_amazon_luxury_time_sage.sh
bash downstream_amazon_luxury_time_sage.sh

# gowalla
# gowalla_Entertainment
bash pretrain_gowalla_entertainment_time_sage.sh
bash downstream_gowalla_entertainment_time_sage.sh
# gowalla_Food
bash pretrain_gowalla_food_time_sage.sh
bash downstream_gowalla_food_time_sage.sh
# gowalla_Shopping
bash pretrain_gowalla_shopping_time_sage.sh
bash downstream_gowalla_shopping_time_sage.sh
```

<!-- # pretrain + amazon_beauty + time transfer: model VGAE
bash pretrain_amazon_beauty_time_vgae.sh
# downstream + amazon_beauty + time transfer: model VGAE
bash downstream_amazon_beauty_time_vgae.sh -->

### Field transfer

```bash
cd scripts/field/
# amazon
bash pretrain_amazon_acs_field_sage.sh
bash downstream_amazon_beauty_field_sage.sh
bash downstream_amazon_fashion_field_sage.sh
bash downstream_amazon_luxury_field_sage.sh

# gowalla
bash pretrain_gowalla_community_field_sage.sh
bash downstream_gowalla_entertainment_field_sage.sh
bash downstream_gowalla_food_field_sage.sh
bash downstream_gowalla_shopping_field_sage.sh
```

### Time+Field transfer

```bash
cd scripts/time_field/
# amazon
bash pretrain_amazon_acs_time_field_sage.sh
bash downstream_amazon_beauty_time_field_sage.sh
bash downstream_amazon_fashion_time_field_sage.sh
bash downstream_amazon_luxury_time_field_sage.sh

# gowalla
bash pretrain_gowalla_community_time_field_sage.sh
bash downstream_gowalla_entertainment_time_field_sage.sh
bash downstream_gowalla_food_time_field_sage.sh
bash downstream_gowalla_shopping_time_field_sage.sh
```