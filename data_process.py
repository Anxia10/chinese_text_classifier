import torch
import pickle as pkl
from tqdm import tqdm
from pathlib import Path

MAX_VOCAB_SIZE = 5000
UNK, PAD = '<UNK>', '<PAD>'


def build_vocab(file_path, language, max_size, min_freq=1):
    vocab_dic = {}
    if language == 'English':
        tokenizer = lambda x: x.split(' ')
    else:
        tokenizer = lambda x: [y for y in x]
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                     :max_size - 2]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


# pkl文件查看
def read_pkl(file_path):
    with open(file_path, 'rb')as f:
        data = pkl.load(f)
        print(data)


# 读取vocab文件得到字典
def read_vocab(vocab_path):
    vocab = {}
    with open(vocab_path,'r',encoding='UTF-8') as f:
        for lin in f:
            # lin = line.strip()
            if not lin:
                continue
            word,index = lin.split('\t')
            vocab[word] = int(index)
    return vocab


# one-hot编码,pad_size:保留的句长
def one_hot(file_path,vocab,language,pad_size=32):
    if language == 'English':
        tokenizer = lambda x: x.split(' ')
    else:
        tokenizer = lambda x: [y for y in x]
    contents = []
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content, label = lin.split('\t')
            words_line = []
            token = tokenizer(content)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, int(label), seq_len))
    return contents

# 输入数据迭代器
class DatasetIterater(object):
    def __init__(self,batches,batch_size,device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数,是整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size:len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


if __name__ == '__main__':
    # 参数信息
    file_path = 'D:/study/Project/word2Vec/THUCNews/data'
    vocab_path = Path(file_path, 'vocab.txt')
    train_path = Path(file_path, 'train.txt')

    # 如果字典不存在，创建字典保存为txt文件
    if vocab_path.exists() is not True:
        vocab = build_vocab(train_path, 'Chinese', 5000)
        vocab = vocab.items()
        with open(vocab_path, 'w', encoding='UTF-8') as f:
            for word, index in vocab:
                f.write(word + '\t' + str(index) + '\n')
    vocab = read_vocab(vocab_path)
    content,_,_ = one_hot(train_path,vocab,'chinese')
    print(content[0])