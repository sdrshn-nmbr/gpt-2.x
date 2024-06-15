#tiny shakespeare dataset
#* wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r') as f:
    text = f.read()
    # data = text[:1000] # first 1,000 characters
    # print (data[:100])
    print(len(text))