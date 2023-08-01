import html
import copy
import csv
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
class Endorse_Dataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_length=2048):
        
        #self.prompt = (
        #        f"Classify this newspaper page as 1 if it contains an editorial article and 0 if it does NOT contain an editorial article:\n{{page}}\n---\nClassification:\n{{label}}{{eos_token}}"
         #   )
        self.prompt = (
                f"Classify this newspaper page as 1 if it contains an editorial article and 0 if it does NOT contain an editorial article:{{page}}{{eos_token}}"
        )

        cols = ['label','accessed','location','date','newspaper','page','url','uid','ocr_text','year']
        df = pd.read_csv("/n/home09/acamara/endorse/data/endorsedata.csv",names=cols,skiprows=1)

        df['label'] = df['label'].map({'N': int(0), 'E': int(1)})
        df['ocr_text'] = df['ocr_text'].apply(html.unescape)
        df['text_length'] = df['ocr_text'].apply(lambda x: len(x))

        def split_text(row, column_name, size):
            text = row[column_name]
            return [text[i:i+size] for i in range(0, len(text), size)]

        # apply the function to each row in the df
        df['ocr_text'] = df.apply(lambda row: split_text(row, 'ocr_text', max_length), axis=1)

        # explode the list into separate rows
        df = df.explode('ocr_text').reset_index(drop=True)
        
        # Set a seed so that the randomness is reproducible
        np.random.seed(0)

        # Create a list of the three partitions
        partitions = ['train', 'val', 'test']

        # Assign each row in the DataFrame to a partition
        df['partition'] = np.random.choice(partitions, size=len(df), p=[0.5, 0.25, 0.25])

        if partition=='train':
            df = df[df['partition'] == 'train']
        if partition=='val':
            df = df[df['partition'] == 'val']
        if partition=='test':
            df = df[df['partition'] == 'test']

        self.texts = df['ocr_text'].tolist() 
        self.labels = df['label'].tolist() 
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
        
        
    def __getitem__(self, idx):
        prompt_ex = self.prompt.format(
            page=self.texts[idx],
            eos_token=self.tokenizer.eos_token_id
        )
        # Padding our input.
        encoded_text = self.tokenizer(
            prompt_ex, 
            padding="max_length",  # Pad to max_length
            truncation=True,  # Truncate if necessary
            max_length=self.max_length,  # Max length to truncate/pad
            return_tensors="pt",  # Return PyTorch tensors
        )

        input_ids = encoded_text['input_ids'].squeeze()
        attention_mask = encoded_text['attention_mask'].squeeze()
        labels = torch.tensor(self.labels[idx], dtype=torch.int64)
         
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        return inputs


