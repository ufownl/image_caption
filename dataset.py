import sys
import multiprocessing
import json
import jieba
import mxnet as mx
from vocab import Vocabulary

def load_dataset(path):
    with open(path + "/annotations.json") as f:
        s = f.read()
    dataset = json.loads(s)
    dataset = [(path + "/images/" + data["image_id"], list(jieba.cut(caption))) for data in dataset for caption in data["caption"]]
    return [(image, caption, len(caption)) for image, caption in dataset]

def load_image(path):
    with open(path, "rb") as f:
        buf = f.read()
    return mx.image.imdecode(buf)

def cook_image(img, fine_size, load_size):
    img = mx.image.resize_short(img, min(load_size))
    img, _ = mx.image.random_crop(img, fine_size)
    return mx.image.color_normalize(
        img.astype("float32") / 255,
        mean = mx.nd.array([0.485, 0.456, 0.406]),
        std = mx.nd.array([0.229, 0.224, 0.225])
    )

def make_vocab(dataset, max_size=sys.maxsize):
    freq = {}
    words = [w for _, caption, _ in dataset for w in caption]
    for w in words:
        if w in freq:
            freq[w] += 1
        else:
            freq[w] = 1
    freq = sorted(freq.items(), key=lambda tup: tup[1], reverse=True)
    return Vocabulary([k for k, _ in freq[:max_size]])

def tokenize(dataset, vocab):
    return [(image, [vocab.word2idx(w) for w in caption], length) for image, caption, length in dataset]

def buckets(dataset, buckets):
    min_len = -1
    for max_len in buckets:
        bucket = [(image, caption, length) for image, caption, length in dataset if length > min_len and length <= max_len]
        min_len = max_len
        if len(bucket) > 0:
            yield bucket, max_len

def batches(dataset, vocab, batch_size, fine_size, load_size, sequence_length, ctx):
    img_path, tgt_tok, tgt_len = zip(*dataset)
    tgt_tok = list(tgt_tok)
    tgt_len = list(tgt_len)
    batches = len(dataset) // batch_size
    if batches * batch_size < len(dataset):
        batches += 1
    with multiprocessing.Pool(multiprocessing.cpu_count() * 2) as p:
        for i in range(batches):
            start = i * batch_size
            imgs = p.map(_CookingWorker(fine_size, load_size), img_path[start: start + batch_size])
            tgt_bat = mx.nd.array(_pad_batch(_add_sent_prefix(tgt_tok[start: start + batch_size], vocab), vocab, sequence_length + 1), ctx=ctx)
            tgt_len_bat = mx.nd.array(tgt_len[start: start + batch_size], ctx=ctx) + 1
            lbl_bat = mx.nd.array(_pad_batch(_add_sent_suffix(tgt_tok[start: start + batch_size], vocab), vocab, sequence_length + 1), ctx=ctx)
            yield mx.nd.concat(*imgs, dim=0).as_in_context(ctx), tgt_bat, tgt_len_bat, lbl_bat

def _add_sent_prefix(batch, vocab):
    return [[vocab.word2idx("<GO>")] + sent for sent in batch]

def _add_sent_suffix(batch, vocab):
    return [sent + [vocab.word2idx("<EOS>")] for sent in batch]

def _pad_batch(batch, vocab, seq_len):
    return [sent + [vocab.word2idx("<PAD>")] * (seq_len - len(sent)) for sent in batch]


class _CookingWorker:
    def __init__(self, fine_size, load_size):
        self._fine_size = fine_size
        self._load_size = load_size

    def __call__(self, img):
        return cook_image(load_image(img), self._fine_size, self._load_size).T.expand_dims(0)


if __name__ == "__main__":
    dataset = load_dataset("data/training")
    print("dataset size: ", len(dataset))
    print("dataset preview: ", dataset[:10])
    vocab = make_vocab(dataset)
    print("vocab size: ", vocab.size())
    dataset = tokenize(dataset, vocab)
    print("tokenize dataset preview: ", dataset[:10])
    print("buckets preview: ", [(len(bucket), seq_len) for bucket, seq_len in buckets(dataset, [8, 16, 32])])
    print("batch preview: ", next(batches(dataset, vocab, 4, (224, 224), (256, 256), 32, mx.cpu())))
