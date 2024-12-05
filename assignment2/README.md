# CSE 143 (Winter 2020) HW 1 data

This folder contains 5 files, a subset of the 1 Billion Word Benchmark's
heldout set.

Specifically, `1b_benchmark.train.tokens` is taken from sections 0-9,
`1b_benchmark.dev.tokens` is taken from sections 10 and 11, and
`1b_benchmark.test.tokens` is taken from sections 12 and 13.

For this assignment, we added two files where the n gram models are created. In each file we have different versions of how the n grams was implemented. The main differences are the addition of smoothing and interpolation, which affects the perplexity scores and the way the model can predict future words in a sentence. 

To recreate this data (download the raw 1 Billion Word Benchmark and generate the split), run:

```
./subsample_1b_benchmark.sh
```

To run the model with smoothing and no interpolation, run:
```
python3 n_grams_part1_and_2.py
```

To run the model with interpolation and smoothing, run:
```
python3 n_grams_part3.py
```