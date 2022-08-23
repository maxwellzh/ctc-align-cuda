# CTC Alignment implemented with CUDA


## What does this module do?

To understand the function of this module, you must acknowledge the **CTC** first. Have a look at the references if you're new to the term **CTC**.

I recently ran into a problem that required to cast sequences with CTC alignment. The process of the algorithm is straight-forward and simple:

1. Remove all consecutive symbols;
2. Remove all blank symbols (commonly we use index `0` to identify that).

e.g. `1 0 1 2 2 0 0 3 0 4 -> 1 1 2 3 4`

However, things become complicated when there're many many long sequences. Nested loop would consume intolerable computing time. Therefore I decided to implement the alignment algo. in CUDA.

## Install

```shell
python setup.py install
# or install locally
# python setup.py develop
```

## Usage

Please refer to the [source code](ctc_align/__init__.py).

## Reference

1. A brief intro to CTC. https://distill.pub/2017/ctc/