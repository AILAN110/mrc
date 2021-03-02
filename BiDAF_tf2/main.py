import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import warnings

warnings.filterwarnings('ignore')
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)
import layers
import preprocess
import numpy as np

print("tf.__version__:", tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def charcnn():
    # tf.sequence_mask
    pass

class BiDAF:

    def __init__(
            self, clen, qlen, emb_size,
            max_features=5000,
            num_highway_layers=2,
            encoder_dropout=0,
            num_decoders=2,
            decoder_dropout=0,
    ):
        """
        双向注意流模型
        :param clen:context 长度
        :param qlen: question 长度
        :param emb_size: 词向量维度
        :param max_features: 词汇表最大数量
        :param num_highway_layers: 高速神经网络的个数 2
        :param encoder_dropout: encoder dropout 概率大小
        :param num_decoders:解码器个数
        :param decoder_dropout: decoder dropout 概率大
        """
        self.clen = clen   #context长度
        self.qlen = qlen   #question长度
        self.max_features = max_features
        self.emb_size = emb_size
        self.num_highway_layers = num_highway_layers
        self.encoder_dropout = encoder_dropout
        self.num_decoders = num_decoders
        self.decoder_dropout = decoder_dropout

    #模型结构
    def build_model(self):
        """
        构建模型
        :return:
        """
        ###################模型定义#######################
        # 1 embedding 层
        # TODO：homework：使用glove word embedding（或自己训练的w2v） 和 CNN char embedding 
        w_cinn = tf.keras.layers.Input(shape=(self.clen,), name='context_word_input')   #单词,shape不含batch_size
        w_qinn = tf.keras.layers.Input(shape=(self.qlen,), name='question__word_input') #单词

        #加载glove向量
        vocab_size=123252   #vocab_size
        embedding_matrix=preprocess.load_glove()
        word_embedding_layer=tf.keras.layers.Embedding(vocab_size,300,weights=[embedding_matrix],trainable=False)
        wc_emb=word_embedding_layer(w_cinn)
        wq_emb=word_embedding_layer(w_qinn)

        #CharCnn
        c_cinn = tf.keras.layers.Input(shape=(self.clen, 20), name='context_word_input')  # char
        c_qinn = tf.keras.layers.Input(shape=(self.qlen, 20), name='question__word_input')  # char
        char_embedding_layer=tf.keras.layers.Embedding(self.max_features,self.emb_size,embeddings_initializer='uniform')
        cc_emb=char_embedding_layer(c_cinn)
        cq_emb=char_embedding_layer(c_qinn)

        cc_emb=tf.reshape(cc_emb,shape=[None,20,self.emb_size])
        cq_emb=tf.reshape(cq_emb,shape=[None,20,self.emb_size])
        conv1d=tf.keras.layers.Conv1D(filters=6,kernel_size=4,padding='same',activation="relu")   #input_shape
        cc_emb=tf.transpose(cc_emb,perm=[0,2,1])
        cq_emb=tf.transpose(cq_emb,perm=[0,2,1])
        cc_emb=conv1d(cc_emb)   #[b*seq_len,6,xx]
        cq_emb=conv1d(cq_emb)
        #最大池化
        cc_emb=tf.transpose(cc_emb,perm=[0,2,1])
        cq_emb=tf.transpose(cq_emb,perm=[0,2,1])
        max_pool_1d=tf.keras.layers.GlobalMaxPooling1D()
        cc_emb=tf.reshape(max_pool_1d(cc_emb),shape=[None,self.clen,6])
        cq_emb=tf.reshape(max_pool_1d(cq_emb),shape=[None,self.qlen,6])
        #concat
        cemb=tf.concat([wc_emb,cc_emb],axis=-1)
        qemb=tf.concat([wq_emb,cq_emb],axis=-1)
        #全连接
        dense_1=tf.keras.layers.Dense(self.emb_size,activation=tf.keras.activations.softmax)
        cemb = dense_1(cemb)
        qemb = dense_1(qemb)
        # cinn = tf.keras.layers.Input(shape=(self.clen,), name='context_input')   #可看作placeholder
        # qinn = tf.keras.layers.Input(shape=(self.qlen,), name='question_input')

        # embedding_layer = tf.keras.layers.Embedding(self.max_features,
        #                                             self.emb_size,
        #                                             embeddings_initializer='uniform',
        #                                             )
        # cemb = embedding_layer(cinn)    #看作tf.nn.embedding_lookup()
        # qemb = embedding_layer(qinn)    # Model方式，下一层在call中包住上一层

        for i in range(self.num_highway_layers):
            """
            使用两层高速神经网络
            """
            highway_layer = layers.Highway(name=f'Highway{i}')   #自定义网络：Layer
            chighway = tf.keras.layers.TimeDistributed(highway_layer, name=f'CHighway{i}')
            qhighway = tf.keras.layers.TimeDistributed(highway_layer, name=f'QHighway{i}')
            cemb = chighway(cemb)    #输入进入
            qemb = qhighway(qemb)

        ## 2. 上下文嵌入层
        # 编码器 双向LSTM
        encoder_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.emb_size,
                recurrent_dropout=self.encoder_dropout,
                return_sequences=True,
                name='RNNEncoder'
            ), name='BiRNNEncoder'
        )

        cencode = encoder_layer(cemb)  # 编码context
        qencode = encoder_layer(qemb)  # 编码question

        # 3.注意流层
        similarity_layer = layers.Similarity(name='SimilarityLayer')  #相似度
        similarity_matrix = similarity_layer([cencode, qencode])

        c2q_att_layer = layers.C2QAttention(name='C2QAttention')
        q2c_att_layer = layers.Q2CAttention(name='Q2CAttention')

        c2q_att = c2q_att_layer(similarity_matrix, qencode)   #代码需补充
        q2c_att = q2c_att_layer(similarity_matrix, cencode)

        # 上下文嵌入向量的生成
        merged_ctx_layer = layers.MergedContext(name='MergedContext')
        merged_ctx = merged_ctx_layer(cencode, c2q_att, q2c_att)

        # 4.模型层
        modeled_ctx = merged_ctx
        for i in range(self.num_decoders):
            decoder_layer = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    self.emb_size,
                    recurrent_dropout=self.decoder_dropout,
                    return_sequences=True,
                    name=f'RNNDecoder{i}'
                ), name=f'BiRNNDecoder{i}'
            )
            modeled_ctx = decoder_layer(merged_ctx)

        # 5. 输出层
        span_begin_layer = layers.SpanBegin(name='SpanBegin')
        span_begin_prob = span_begin_layer([merged_ctx, modeled_ctx])

        span_end_layer = layers.SpanEnd(name='SpanEnd')
        span_end_prob = span_end_layer([cencode, merged_ctx, modeled_ctx, span_begin_prob])

        output_layer = layers.Combine(name='CombineOutputs')
        out = output_layer([span_begin_prob, span_end_prob])  #最终输出

        # inn = [cinn, qinn]   #输入
        inn = [c_cinn,w_cinn, c_qinn,w_qinn]   #输入

        self.model = tf.keras.models.Model(inn, out)   #固定：输入、输出（fit时数据要对应,多任务out也可以是list），代替Sequential
        self.model.summary(line_length=128)    #输出各层参数状况：类似tf 1.x的summary
        ###############模型编译######################
        optimizer = tf.keras.optimizers.Adadelta(lr=1e-2)
        self.model.compile(
            optimizer=optimizer,   #优化器
            loss=negative_avg_log_error,    #计算loss,多任务可以用list,可以设置loss_weights=[loss1权重，loss2权重。。。]
            metrics=[accuracy]   #评估指标
        )


def negative_avg_log_error(y_true, y_pred):
    """
    损失函数计算
    -1/N{sum(i~N)[(log(p1)+log(p2))]}
    :param y_true:
    :param y_pred:
    :return:
    """

    def sum_of_log_prob(inputs):
        y_true, y_pred_start, y_pred_end = inputs

        begin_idx = tf.dtypes.cast(y_true[0], dtype=tf.int32)
        end_idx = tf.dtypes.cast(y_true[1], dtype=tf.int32)

        begin_prob = y_pred_start[begin_idx]
        end_prob = y_pred_end[end_idx]

        return tf.math.log(begin_prob) + tf.math.log(end_prob)

    y_true = tf.squeeze(y_true)
    y_pred_start = y_pred[:, 0, :]
    y_pred_end = y_pred[:, 1, :]

    inputs = (y_true, y_pred_start, y_pred_end)
    batch_prob_sum = tf.map_fn(sum_of_log_prob, inputs, dtype=tf.float32)

    return -tf.keras.backend.mean(batch_prob_sum, axis=0, keepdims=True)


def accuracy(y_true, y_pred):
    """
    准确率计算
    :param y_true:
    :param y_pred:
    :return:
    """

    def calc_acc(inputs):
        y_true, y_pred_start, y_pred_end = inputs

        begin_idx = tf.dtypes.cast(y_true[0], dtype=tf.int32)
        end_idx = tf.dtypes.cast(y_true[1], dtype=tf.int32)

        start_probability = y_pred_start[begin_idx]
        end_probability = y_pred_end[end_idx]

        return (start_probability + end_probability) / 2.0

    y_true = tf.squeeze(y_true)
    y_pred_start = y_pred[:, 0, :]
    y_pred_end = y_pred[:, 1, :]

    inputs = (y_true, y_pred_start, y_pred_end)
    acc = tf.map_fn(calc_acc, inputs, dtype=tf.float32)

    return tf.math.reduce_mean(acc, axis=0)


if __name__ == '__main__':
    ds = preprocess.Preprocessor([
        './data/squad/train-v1.1.json',
        './data/squad/dev-v1.1.json',
        './data/squad/dev-v1.1.json'
    ])
    train_c, train_cw, train_q, train_qw, train_y = ds.get_dataset('./data/squad/train-v1.1.json')  #context_id、question_id、answer位置
    test_c, test_q, test_y = ds.get_dataset('./data/squad/dev-v1.1.json')

    print(train_c.shape, train_q.shape, train_y.shape)
    print(test_c.shape, test_q.shape, test_y.shape)

    bidaf = BiDAF(
        clen=ds.max_clen,
        qlen=ds.max_qlen,
        emb_size=50,
        max_features=len(ds.charset)
    )
    bidaf.build_model()   #模型结构
    bidaf.model.fit(                  #模型训练
        [train_c,train_cw, train_q,train_qw],          #输入
        train_y,                     #输出label来求loss
        batch_size=64,
        epochs=10,
        validation_data=([test_c, test_q], test_y)       #验证集
    )
