# AI/ML tools for model inspection


### llama.cpp/src/llama.cpp
a modified version of `llama.cpp` (from commit: `faac0bae265449fd988c57bf894018edc36fbe1e`) that prints tensor names to stdout upon model start. Useful to inspect the model. The original `llama.cpp` file is saved as `llama.cpp.orig`.


### compare_tensors.py
a program to compare the names of tensors between PyTorch and GGUF models, to double check on format conversions

