import math
import argparse
import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt
from vocab import Vocabulary
from dataset import load_image, cook_image
from img2seq import FeatureExtractor, Feature2seqTransformer


def visualize(img):
    mean = mx.nd.array([0.485, 0.456, 0.406])
    std = mx.nd.array([0.229, 0.224, 0.225])
    plt.imshow(((img * std + mean) * 255).asnumpy().astype(np.uint8))
    plt.axis("off")


parser = argparse.ArgumentParser(description="Start a ai_challenger_caption tester.")
parser.add_argument("images", metavar="IMG", help="path of the image file[s]", type=str, nargs="+")
parser.add_argument("--beam", help="using beam search", action="store_true")
parser.add_argument("--beam_size", help="set the size of beam (default: 10)", type=int, default=10)
parser.add_argument("--visualize", help="visualizing the images", action="store_true")
parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
args = parser.parse_args()

fine_size = (224, 224)
load_size = (256, 256)
feature_length = 49
sequence_length = 32
beam_size = 10
if len(args.images) <= 4:
    rows = 1
    cols = len(args.images)
else:
    rows = math.ceil(len(args.images) / 4)
    cols = 4
if args.gpu:
    context = mx.gpu(args.device_id)
else:
    context = mx.cpu(args.device_id)

print("Loading vocabulary...", flush=True)
vocab = Vocabulary()
vocab.load("model/vocabulary.json")

print("Loading model...", flush=True)
features = FeatureExtractor(ctx=context)
model = Feature2seqTransformer(feature_length, vocab.size(), sequence_length)
model.load_parameters("model/img2seq.params", ctx=context)

index = 0
for path in args.images:
    index += 1
    print(path)
    image = cook_image(load_image(path), fine_size, load_size)
    if args.visualize:
        plt.subplot(rows, cols, index)
        visualize(image)
    image = image.T.expand_dims(0).as_in_context(context)
    source = features(image)
    enc_out, enc_self_attn = model.encode(source)

    if args.beam:
        sequences = [([vocab.word2idx("<GO>")], 0.0)]
        while True:
            candidates = []
            for seq, score in sequences:
                if seq[-1] == vocab.word2idx("<EOS>") or len(seq) >= source.shape[1] + 2:
                    candidates.append((seq, score))
                else:
                    target = mx.nd.array(seq, ctx=context).reshape((1, -1))
                    tgt_len = mx.nd.array([len(seq)], ctx=context)
                    output, dec_self_attn, context_attn = model.decode(target, tgt_len, enc_out)
                    probs = mx.nd.softmax(output, axis=2)
                    beam = probs[0, -1].topk(k=beam_size, ret_typ="both")
                    for i in range(beam_size):
                        candidates.append((seq + [int(beam[1][i].asscalar())], score + math.log(beam[0][i].asscalar())))
            if len(candidates) <= len(sequences):
                break;
            sequences = sorted(candidates, key=lambda tup: tup[1], reverse=True)[:beam_size]

        scores = mx.nd.array([score for _, score in sequences], ctx=context)
        probs = mx.nd.softmax(scores)

        for i, (seq, score) in enumerate(sequences):
            text = ""
            for token in seq[1:-1]:
                text += vocab.idx2word(token)
            print(text, score, probs[i].asscalar())
            print(seq)
    else:
        sequence = [vocab.word2idx("<GO>")]
        target = mx.nd.array(sequence, ctx=context).reshape((1, -1))
        tgt_len = mx.nd.array([len(sequence)], ctx=context)
        while True:
            output, dec_self_attn, context_attn = model.decode(target, tgt_len, enc_out)
            index = mx.nd.argmax(output, axis=2)
            word_token = index[0, -1].asscalar()
            sequence += [word_token]
            if word_token == vocab.word2idx("<EOS>") or len(sequence) >= source.shape[1] + 2:
                break;
            target = mx.nd.array(sequence, ctx=context).reshape((1, -1))
            tgt_len = mx.nd.array([len(sequence)], ctx=context)
            print(vocab.idx2word(word_token), end="", flush=True)
        print("") 
        print(sequence)

if args.visualize:
    plt.show()
