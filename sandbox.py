#tiny shakespeare dataset
#* wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# with open('fineweb.txt', 'r', encoding="utf-8") as f:
#     text = f.read()
#     data = text[:1000 - 1] # first 1,000 characters
#     # print (data[:100])
#     # print(len(text))

# import tiktoken
# import torch
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode(data)
# print(tokens[:24])

# buf = torch.tensor(tokens[:24 + 1])
# x = buf[:-1].view(4, 6) # 4 * 6 = 24
# print(x)
# labels = buf[1:].view(4, 6)
# print(labels)

with open ('fineweb.txt', 'r', encoding="utf-8") as f:
    text = f.read()
    
with open ('fineweb_sample.txt', 'w', encoding="utf-8") as f:
    f.write(text[:1000000])

# gt_tokens = torch.full_like()