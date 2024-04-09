#### Experimental procedure

```cmd
step1: bash download.sh download plm checkpoint
step2: bash train_with_[bert/bge/electra/roberta].sh
```

#### Results

| Model            | LAS | UAS |
|------------------|-----------|-----------|
| Electra+Biaffine | 0.8321    | 0.8731    |
| Roberta+Biaffine | 0.8213    | 0.8320    |
| Bert+Biaffine    | 0.8019    | 0.8243    |
| Bge+Biaffine     | 0.8710    | 0.9012    |