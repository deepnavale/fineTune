# Finetuning the Gemma 2B Model

This project demonstrates how to finetune Google's Gemma 2B model using the TRL (Transformer Reinforcement Learning) library and LoRA (Low-Rank Adaptation) for efficient training. The notebook uses the Hugging Face ecosystem to load the model and dataset and is designed to be run in a Google Colab environment.

---

### Prerequisites :gear:

To run this notebook, you will need to install the following libraries:

* `bitsandbytes` (version 0.42.0)
* `peft` (version 0.8.2)
* `trl` (version 0.7.10)
* `accelerate` (version 0.27.1)
* `datasets` (version 2.17.0)
* `transformers` (version 4.38.0)

You must also have a Hugging Face token. This is required to access the Gemma model and dataset from the Hugging Face Hub.

---

### How to Run :rocket:

1.  **Set Up Your Environment:**
    * Open the `Gemma_Finetuning_notebook.ipynb` file in a Jupyter environment like Google Colab.
    * Install the required libraries by running the `!pip3 install ...` cell.

2.  **Authenticate with Hugging Face:**
    * Set your Hugging Face token as an environment variable. The notebook includes a placeholder `os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN_HERE"`. **Replace `YOUR_HF_TOKEN_HERE` with your actual token.**

3.  **Run the Notebook:**
    * Execute the cells in the notebook sequentially. The notebook will automatically perform the following steps:
        * Load the Gemma 2B model with 4-bit quantization.
        * Load the `Abirate/english_quotes` dataset from the Hugging Face Hub.
        * Configure and apply LoRA for finetuning.
        * Initialize the `SFTTrainer` for supervised finetuning.
        * Start the training process.

---

### Model Details :robot:

* **Base Model:** `google/gemma-2b`
* **Quantization:** The model is loaded in 4-bit using `BitsAndBytesConfig` to reduce memory usage.
* **Finetuning Method:** LoRA is used for efficient parameter-efficient finetuning.

---

### Dataset :open_file_folder:

The notebook uses the `Abirate/english_quotes` dataset, which contains English quotes and their authors. A formatting function is defined to structure the data for training in the format "Quote: [quote]\nAuthor: [author]".

---

### Training :chart_with_upwards_trend:

The `SFTTrainer` is configured with the following training arguments:

* **Batch Size:** `per_device_train_batch_size=1`
* **Gradient Accumulation:** `gradient_accumulation_steps=4`
* **Steps:** `max_steps=100`
* **Learning Rate:** `learning_rate=2e-4`
* **Precision:** `fp16=True`
* **Optimizer:** `paged_adamw_8bit`
