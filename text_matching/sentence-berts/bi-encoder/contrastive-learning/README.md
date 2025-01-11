# Data

Prepare the train.json/dev.json/test.txt under the data directory.

# Train

`data/train.json`:
```
{"query": "anchor sentence", "pos": "positive sentence", "neg": ["negative sentence1", "negative sentence2", ...]}
{"query": "anchor sentence", "pos": "positive sentence", "neg": ["negative sentence1", "negative sentence2", ...]}
```

Change arguments in `train.sh`.

```sh
sh train.sh
```

# Test
The test data file is constructed with two sentences separated by a tab (`\t`), and there is a numerical value indicating whether they are similar.
`data/test.txt`:
```
Hello	Hi	1
Nice to see you.	Nice	0
```


Change `model_name_or_path` in `test.sh` as output of `train.py`.
```sh
sh test.sh
```



# Acknowledgement
The code is developed with reference to [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding), [uniem](https://github.com/wangyuxinwhy/uniem), and [sentence-transformers](https://github.com/UKPLab/sentence-transformers).



