# Data
The data file is constructed with two sentences separated by a tab (`\t`), and there is a numerical value indicating whether they are similar.

`data/train.txt`:
```
Hello	Hi	1
Nice to see you.	Nice	0
```

Prepare the train.txt/test.tx/dev.txt under the data dir.

# Train

Change arguments in `train.sh`.

```sh
sh train.sh
```

# Test

Change `model_name_or_path` in `test.sh` as output of `train.py`.
```sh
sh test.sh
```



# Acknowledgement
The code is developed with reference to [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding), [uniem](https://github.com/wangyuxinwhy/uniem), and [sentence-transformers](https://github.com/UKPLab/sentence-transformers).



