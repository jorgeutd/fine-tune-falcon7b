# Fine-tuning falcon-7b with QLoRA

This repository contains a demo notebook that demonstrates the fine-tuning of the falcon-7b model using QLoRA, Hugging Face PEFT, and bitsandbytes.

## Requirements

The notebook was tested in Amazon SageMaker Studio with Python 3 (Data Science 3.0) kernel with a ml.g5.8xlarge instance. You'll need to install the following libraries:

- torch==2.0.1
- bitsandbytes==0.39.1
- datasets
- py7zr
- einops
- tensorboardX
- transformers (latest from git)
- peft (latest from git)
- accelerate (latest from git)

You can install these libraries using pip:

```bash
pip install -q -U torch==2.0.1 bitsandbytes==0.39.1
pip install -q -U datasets py7zr einops tensorboardX
pip install -q -U git+https://github.com/huggingface/transformers.git@850cf4af0ce281d2c3e7ebfc12e0bc24a9c40714
pip install -q -U git+https://github.com/huggingface/peft.git@e2b8e3260d3eeb736edf21a2424e89fe3ecf429d
pip install -q -U git+https://github.com/huggingface/accelerate.git@b76409ba05e6fa7dfc59d50eee1734672126fdba
```

## Notebook Description

The notebook involves the following steps:

1. **Preparation**: Load the necessary libraries and set up the CUDA environment for bitsandbytes.
2. **Model Setup**: Load the falcon-7b model using the Hugging Face Transformers library and prepare the model for 4-bit quantization using bitsandbytes. Then, prepare the model for QLoRA training using PEFT.
3. **Dataset Loading**: Load the SAMSum dataset using the Hugging Face Datasets library.
4. **Data Preprocessing**: Apply prompt templates to the dataset samples and tokenize the dataset.
5. **Training Setup**: Create a Trainer instance from the Hugging Face Transformers library with the specified hyperparameters and start the training process.
6. **Model Evaluation**: Evaluate the model and print the evaluation metrics.
7. **Inference**: Load a test dataset, select a random sample, and generate a summary for it using the fine-tuned model.

Please refer to the notebook for detailed code and explanations.

