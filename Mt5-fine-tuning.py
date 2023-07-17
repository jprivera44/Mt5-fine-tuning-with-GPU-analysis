#Script that runs the Fine tuning separetely in order to create CUDA statistics
#Needs to be run separate in order to not take too much GPU memory.
#Author: JP Rivera


import torch
import pickle
import argparse
from transformers import TrainingArguments, Trainer
import numpy as np
from transformers import MT5ForConditionalGeneration, Trainer, TrainingArguments

# Define the compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}


def main(model_path: str, train_data_path: str, val_data_path: str, training_args_path: str):
    #model = torch.load(model_path)  # Load the model
    #model = AutoModel.from_pretrained(model_path)
    model = MT5ForConditionalGeneration.from_pretrained(model_path)
    training_args = torch.load(training_args_path)  # Load the training arguments
    
    

    with open(train_data_path, 'rb') as f:
        model_train_data = pickle.load(f)

    with open(val_data_path, 'rb') as f:
        model_val_data = pickle.load(f)
        
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=model_train_data,
        eval_dataset=model_val_data,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()



#Main method acctps inpus args
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--val_data_path", type=str)
    parser.add_argument("--training_args_path", type=str)
    args = parser.parse_args()
    main(args.model_path, args.train_data_path, args.val_data_path, args.training_args_path)

   