import mxnet as mx
from transformer_utils import PositionalEncoding, TimingEncoding, EncoderLayer, AdaptiveComputationTime, Decoder

class FeatureExtractor:
    def __init__(self, ctx=mx.cpu()):
        net = mx.gluon.model_zoo.vision.resnet34_v2(pretrained=True, ctx=ctx)
        self._features = mx.gluon.nn.HybridSequential()
        with self._features.name_scope():
            for block in net.features[:11]:
                self._features.add(block)
        self._features.hybridize()

    def __call__(self, inputs):
        f = self._features(inputs)
        return f.reshape((f.shape[0], f.shape[1], -1)).transpose(axes=(0, 2, 1))


class FeatureEncoder(mx.gluon.nn.Block):
    def __init__(self, feature_length, layers, dims, heads, ffn_dims, dropout=0.0, **kwargs):
        super(FeatureEncoder, self).__init__(**kwargs)
        with self.name_scope():
            self._pos_encoding = PositionalEncoding(dims, feature_length)
            self._time_encoding = TimingEncoding(dims, layers)
            self._encoder = EncoderLayer(dims, heads, ffn_dims, dropout)
            self._act = AdaptiveComputationTime(layers)

    def forward(self, x):
        seq_len = mx.nd.array([x.shape[1]] * x.shape[0], ctx=x.context)
        return self._act(self._encoder, self._pos_encoding, self._time_encoding, x, seq_len, None)


class Feature2seqTransformer(mx.gluon.nn.Block):
    def __init__(self, feature_length, vocab_size, sequence_length, layers=6, dims=512, heads=8, ffn_dims=2048, dropout=0.2, **kwargs):
        super(Feature2seqTransformer, self).__init__(**kwargs)
        with self.name_scope():
            self._encoder = FeatureEncoder(feature_length, layers, dims, heads, ffn_dims, dropout)
            self._decoder = Decoder(vocab_size, sequence_length + 1, layers, dims, heads, ffn_dims, dropout)
            self._output = mx.gluon.nn.Dense(vocab_size, flatten=False)

    def forward(self, features, tgt_seq, tgt_len):
       out, enc_self_attn = self.encode(features) 
       out, dec_self_attn, context_attn = self.decode(tgt_seq, tgt_len, out)
       return out, enc_self_attn, dec_self_attn, context_attn

    def encode(self, features):
        return self._encoder(features)

    def decode(self, seq, seq_len, enc_out):
        out, self_attn, context_attn = self._decoder(seq, seq_len, enc_out, None)
        out = self._output(out)
        return out, self_attn, context_attn
        

if __name__ == "__main__":
    features = FeatureExtractor()
    print(features(mx.nd.zeros((4, 3, 224, 224))))
    model = Feature2seqTransformer(49, 128, 32)
    model.initialize(mx.init.Xavier())
    print(model(features(mx.nd.zeros((4, 3, 224, 224))), mx.nd.zeros((4, 8)), mx.nd.ones((4,)) * 8))
