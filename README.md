# pytorch_cppcuda_practice
Practice to write cpp/cuda extension for pytorch

This simple example writes a custom matrix addition in cuda, and provides python binding with forward/backward operations.

Sources: [1](https://qiita.com/windfall/items/5031d70c649b06a1534f) [2](https://pytorch.org/tutorials/advanced/cpp_extension.html)

# Installation

1.  Install pytorch.
2.  From this repo, run `pip install . --use-feature=in-tree-build`

# Test

Run `python test.py`.

# Misc

For vscode to resolve `<torch/extension.h>`, add the following lines to the include path (change python path and version):
```
"<path to your conda environment>/include/python3.8",
"<path to your conda environment>/lib/python3.8/site-packages/torch/include",
"<path to your conda environment>/lib/python3.8/site-packages/torch/include/torch/csrc/api/include"
```
