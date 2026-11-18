import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Any

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        # fixed length every encoder/decoder sequence is padded to, so sentences can be batched together
        self.seq_len = seq_len

        # special token IDs, wrapped in 1-element tensors so they can be torch.cat'd onto the sentence tokens.
        # torch.tensor (lowercase) is used because torch.Tensor doesn't accept a dtype argument.
        # why int64? token IDs are indices into the embedding table, and nn.Embedding expects
        # integer (long, i.e. int64) indices, not floats.
        # using tokenizer_src for these assumes both tokenizers assign the same IDs to special tokens.
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype = torch.int64)


    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # text -> list of token IDs; each language uses its own tokenizer
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # encoder input = [SOS] + sentence + [EOS] + padding -> reserve 2 slots for special tokens
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        # decoder input = [SOS] + sentence + padding -> reserve 1 slot.
        # (the label is sentence + [EOS] + padding, also 1 slot: it's the decoder input shifted left by one,
        #  so the decoder learns to predict the next token at each position)
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        
        # Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64)
            ]
        )

        # Add SOS to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64) 
            ]
        )

        # Add EOS to the label ( what we expect as output from the decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input" : encoder_input, # (Seq_len)
            "decoder_input": decoder_input, # (Seq_len)
            # padding mask: 1 = real token, 0 = [PAD]. (Seq_len) -> (1, 1, Seq_len) so it broadcasts
            # over (heads, query positions) inside attention scores of shape (B, h, Seq_len, Seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, Seq_len)
            # padding mask AND causal mask: each position may only attend to real tokens at or before it
            # (1, 1, Seq_len) & (1, Seq_len, Seq_len) -> (1, Seq_len, Seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, Seq_len, Seq_len)
            "label": label, #(Seq_len)    
            "src_text": src_text,
            "tgt_text": tgt_text 
        }

def causal_mask(size):
    # upper triangle above the diagonal = future positions -> 1; then flip so allowed positions are True
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int) # (1, Seq_len, Seq_len)
    return mask == 0 # True where query position i may attend to key position j (j <= i)
