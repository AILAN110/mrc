# -*- coding:utf-8 -*-
import numpy as np
import data_io as pio
from nltk.corpus import stopwords
import nltk
import json
import os
stops=stopwords.words("english")
#加载glove词向量
def load_glove(word_path="data/vocab/words.txt",glove_path="data/vocab/glove.6B.300d.txt"):
    '''
    :param glove_path:词向量路径
    :param words: 单词list
    :return:
    '''
    words=load_pkl(word_path)
    #加载词向量
    with open(glove_path, "r", encoding="utf-8") as glove_f:
        all_vectors = np.zeros((len(words),300))
        #0-unk,1-sep,2-cls,3-pad
        word_vector_dict = {}
        ok, unk = 0, 0
        for line in glove_f:
            segs = line.strip().split()
            assert len(segs) == 301
            word_vector_dict[segs[0]] = [float(word) for word in segs[1:]]

        for i in range(len(words)):
            if words[i] in word_vector_dict:
                all_vectors[i]=word_vector_dict[words[i]]
                ok += 1
            else:
                # all_vectors[i]=word_vector_dict['<unk>']
                unk += 1
        print("ok={},unk={}".format(ok,unk))
        print(all_vectors.shape)
        return all_vectors

def dump_pkl(dataset,f_path="data/vocab/words.txt"):
    with open(f_path,'w',encoding="utf-8") as fw:
        json.dump(dataset,fw,ensure_ascii=False)

def load_pkl(f_path="data/vocab/words.txt"):
    with open(f_path,"r",encoding="utf-8") as w_r:
        dataset=json.load(w_r)
    return dataset

#字母级别
class Preprocessor:
    def __init__(self, datasets_fp, max_length=384, stride=128):
        self.datasets_fp = datasets_fp
        self.max_length = max_length
        self.max_clen = 100
        self.max_qlen = 100
        self.stride = stride
        self.wordset=set()     #单词集
        self.charset = set()   #char集
        self.build_charset()   #建立char字典
        self.build_wordset()   #建立word字典

    #构建words集
    def build_wordset(self,f_path="data/vocab/words.txt"):
        if os.path.exists(f_path):
            self.wordset=load_pkl(f_path)
        else:
            for fp in self.datasets_fp:
                dataset=pio.load(fp)
                for _, context, question, answer, _ in self.iter_cqa(dataset):
                    words=[]
                    words.extend(nltk.word_tokenize(context))
                    words.extend(nltk.word_tokenize(question))
                    words.extend(nltk.word_tokenize(answer))
                    self.wordset |= set(words)
            self.wordset=[w for w in self.wordset if w not in stops]  #去停用词
            self.wordset = sorted(self.wordset)  # 按字母训练排序(偷懒)
            self.wordset = ['[PAD]', '[CLS]', '[SEP]','[UNK]'] + self.wordset
            dump_pkl(self.wordset,f_path)
        print(len(self.wordset))
        idx = list(range(len(self.wordset)))
        self.word2id = dict(zip(self.wordset, idx))
        self.id2word = dict(zip(idx, self.charset))

    #建立id2token/token2id
    def build_charset(self):
        for fp in self.datasets_fp:
            self.charset |= self.dataset_info(fp)

        self.charset = sorted(list(self.charset))   #按字母训练排序
        self.charset = ['[PAD]', '[CLS]', '[SEP]'] + self.charset + ['[UNK]']
        idx = list(range(len(self.charset)))
        self.ch2id = dict(zip(self.charset, idx))
        self.id2ch = dict(zip(idx, self.charset))
        # print(self.ch2id, self.id2ch)

    #所有token整合去重
    def dataset_info(self, inn):
        charset = set()
        dataset = pio.load(inn)   #加载数据

        for _, context, question, answer, _ in self.iter_cqa(dataset):
            charset |= set(context) | set(question) | set(answer)   #分字去重
            # self.max_clen = max(self.max_clen, len(context))
            # self.max_qlen = max(self.max_clen, len(question))

        return charset
    #解析json,迭代器
    def iter_cqa(self, dataset):
        for data in dataset['data']:
            for paragraph in data['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    qid = qa['id']
                    question = qa['question']
                    for answer in qa['answers']:
                        text = answer['text']
                        answer_start = answer['answer_start']
                        yield qid, context, question, text, answer_start

    #question_id context_id
    def encode(self, context, question):
        question_encode = self.convert2id(question, begin=True, end=True)   #question填充特殊token
        left_length = self.max_length - len(question_encode)  #长度
        context_encode = self.convert2id(context, maxlen=left_length, end=True)  #question填充特殊token
        cq_encode = question_encode + context_encode

        assert len(cq_encode) == self.max_length

        return cq_encode
    #填充[CLS]、[SEP]、[PAD],并转成id
    def convert2id(self, sent, maxlen=None, begin=False, end=False):
        ch = [ch for ch in sent]    #分字
        ch = ['[CLS]'] * begin + ch

        if maxlen is not None:
            ch = ch[:maxlen - 1 * end]
            ch += ['[SEP]'] * end
            ch += ['[PAD]'] * (maxlen - len(ch))
        else:
            ch += ['[SEP]'] * end

        ids = list(map(self.get_id, ch))

        return ids
    #token 转 id
    def get_id(self, ch):
        return self.ch2id.get(ch, self.ch2id['[UNK]'])

    #一条数据：段落id、提问id、(开始、结束)
    def get_dataset(self, ds_fp,w_type="word"):
        cs, qs, be = [], [], []
        cc,qc=list(),list()   #用于做conv1d
        for _, c, cw, q, qw, b, e in self.get_data(ds_fp,w_type):
            if w_type=="word":
                cc.append([w for w in self.get_sen_char(cw)])
                qc.append([w for w in self.get_sen_char(qw)])
            cs.append(c)   #上下文，二维
            qs.append(q)   #提问，二维
            be.append((b, e))  #开始、结束
        return map(np.array, (cs,cc,qs,qc,be))

    #id格式数据：迭代器
    def get_data(self, ds_fp,w_type):
        dataset = pio.load(ds_fp)   #加载文件，根据后缀指定加载方式
        for qid, context, question, text, answer_start in self.iter_cqa(dataset):
            cids,cw = self.get_sent_ids(context, self.max_clen,w_type=w_type)   #转成格式化id,一维
            qids,qw = self.get_sent_ids(question, self.max_qlen,w_type=w_type)  #转成格式化id
            # print(cids)
            b, e = answer_start, answer_start + len(text)   #开始位置，结束位置
            if e >= len(cids):
                b = e = 0
            yield qid, cids,cw, qids,qw, b, e
    #新增
    def word_convert2id(self, sent, maxlen=None, begin=False, end=False):
        def get_id(wh):
            return self.word2id.get(wh, self.word2id['[UNK]'])
        wh = [wh for wh in nltk.word_tokenize(sent)]  # 分字
        wh = ['[CLS]'] * begin + wh

        if maxlen is not None:
            wh = wh[:maxlen - 1 * end]
            wh += ['[SEP]'] * end
            wh += ['[PAD]'] * (maxlen - len(wh))
        else:
            wh += ['[SEP]'] * end

        ids = list(map(get_id, wh))
        return ids,wh

    #新增
    def get_sen_char(self,words,max_len=20,padding='[PAD]'):
        '''
        :param words: 一行list
        :param max_len: 最长20个字母
        :param padding:
        :return:二维
        '''
        res=[]
        # print(words)
        for word in words:
            if word in ['[PAD]','[SEP]','[CLS]','[UNK]']:
                w=[self.get_id(word)]
                w+=[self.get_id(padding)]*(max_len-1)
            else:
                w=[self.get_id(w_) for w_ in word.strip()]
                w+=[self.get_id(padding)]*(max_len-len(w))
            res.append(w[:max_len])
        # print(type(res))
        return res

    # 填充[CLS]、[SEP]、[PAD],并转成id
    def get_sent_ids(self, sent, maxlen,w_type="char"):
        if w_type=="char":
            return self.convert2id(sent, maxlen=maxlen, end=True),[""]*maxlen
        else:
            return self.word_convert2id(sent,maxlen,end=True)

if __name__ == '__main__':
    # p = Preprocessor([
    #     './data/squad/train-v1.1.json',
    #     './data/squad/dev-v1.1.json',
    #     './data/squad/dev-v1.1.json'
    # ])   #建立字典
    # print(p.wordset)
    # a,aa,b,bb,c=p.get_dataset('./data/squad/dev-v1.1.json')
    # print(p.encode('modern stone statue of Mary', 'To whom did the Virgin Mary '))
    matrx=load_glove()
    print(matrx)
