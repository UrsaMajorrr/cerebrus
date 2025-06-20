import torch
from torch.utils.data import Dataset, DataLoader 
import tiktoken

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length=512, stride=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.txt = txt

        self.input_ids = []
        self.output_ids = []

        token_ids = self.tokenizer.encode(self.txt)

        for i in range(0, len(token_ids) - self.max_length, self.stride):
            input_ids = token_ids[i:i+self.max_length]
            target_ids = token_ids[i+1:i+self.max_length+1]
            self.input_ids.append(torch.tensor(input_ids))
            self.output_ids.append(torch.tensor(target_ids))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.output_ids[idx]
    
def create_dataloader(txt, max_length=512, stride=128, batch_size=16, shuffle=True, num_workers=0, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)

with open("the-verdict.txt", "r", encoding="utf-8") as file:
    txt = file.read()

dataloader = create_dataloader(txt, max_length=4, stride=4, batch_size=1, shuffle=False)
data_iter = iter(dataloader)
print(next(data_iter)) 
print(next(data_iter)) 