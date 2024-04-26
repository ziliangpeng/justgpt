"""
Trains a GPT to add n-digit numbers.

    python simpleadder.py --data.ndigit=3


it seems ok to get to 6 digits using gpt-nano (occasionally stuck), start to get stuck a lot at 7 digits.
using gpt-micro makes 7 digits better.
"""

import math
import os
import random
import sys
import json

import numpy

from datadog import statsd
from prometheus_client import Gauge, start_http_server

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------

# metmul. will move out.

def has_datadog():
    # return os.environ.get("METMUL_DATADOG", False)
    return True

def has_prometheus():
    # return os.environ.get("METMUL_PROMETHEUS", False)
    return True

prom_gauge_dict = {}

if has_prometheus():
    try:
        start_http_server(9090)
    except Exception as e:
        print("Error in acquiring prometheus port. Process ID:", os.getpid())

def gauge(name, value, tags=None):
    if has_datadog():
        statsd.gauge(name, value, tags=tags)
    if has_prometheus():
        name = name.replace(".", "_")
        label_kv = {kv.split(":")[0]: kv.split(":")[1] for kv in tags if ":" in kv}
        all_tag_key = label_kv.keys()
        gauge_dict_key = name + '@' + str(sorted(all_tag_key))
        if gauge_dict_key not in prom_gauge_dict:
            prom_gauge_dict[gauge_dict_key] = Gauge(name, 'auto generated: ' + name, all_tag_key)
        prom_gauge_dict[gauge_dict_key].labels(**label_kv).set(value)

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/adder'

    # data
    C.data = AdditionDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-nano'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 3e-4 # the model we're using is so small that we can go a bit faster
    C.trainer.max_iters = 100000
    C.trainer.batch_size = 64

    return C

# -----------------------------------------------------------------------------

class AdditionDataset(Dataset):
    """
    Creates n-digit addition problems. For example, if n=2, then an example
    addition problem would be to add 85 + 50 = 135. This problem would be
    represented as the following string for the GPT:

    "8550531"

    This is because:
    - we are discarding the + and =, which are not necessary. We just encode the digits
      of the input numbers concatenated together.
    - the result 135 is encoded backwards to make the addition easier to learn for the
      GPT model, because of how the addition algorithm works.

    As one more example, the problem 6 + 39 = 45 would be encoded as:

    "0639054"

    where you will notice that we are padding with zeros to make sure that we always
    produce strings of the exact same size: n + n + (n + 1). When n=2, this is 7.
    At test time, we will feed in an addition problem by giving the first 2n digits,
    and hoping that the GPT model completes the sequence with the next (n+1) digits
    correctly.
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.ndigit = 7
        return C

    def __init__(self, config, split):
        self.config = config
        self.split = split # train/test

        # split up all addition problems into either training data or test data
        ndigit = self.config.ndigit

    def get_vocab_size(self):
        return 10 # digits 0..9

    def get_block_size(self):
        # a,b,a+b, and +1 due to potential carry overflow,
        # but then also -1 because very last digit doesn't ever plug back
        # as there is no explicit <EOS> token to predict, it is implied
        return 3*self.config.ndigit + 1 - 1

    def __len__(self):
        if self.split == 'test':
            return 500
        else:
            # l must be int32 otherwise it breaks torch.
            l = (10 ** self.config.ndigit) ** 2
            if l > 2000000000:
                l = 2000000000
            return l

    def __getitem__(self, idx):
        ndigit = self.config.ndigit
        # given a problem index idx, first recover the associated a + b
        # TODO: with pure randomness, train and test will overlap. We need to ensure a clean separation, but first let the mixed data suceed.
        idx = random.randint(0, (10 ** self.config.ndigit) ** 2 - 1)
        nd = 10**ndigit
        a = idx // nd
        b = idx %  nd
        # calculate the "label" of the addition problem a + b
        c = a + b
        # encode the digits of a, b, c into strings
        astr = f'%0{ndigit}d' % a
        bstr = f'%0{ndigit}d' % b
        cstr = (f'%0{ndigit+1}d' % c)[::-1] # reverse c to make addition easier
        render = astr + bstr + cstr
        dix = [int(s) for s in render] # convert each character to its token index
        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long) # predict the next token in the sequence
        y[:ndigit*2-1] = -1 # we will only train in the output locations. -1 will mask loss to zero
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # first of all, float16 will give a nan loss, due to lack of range.
    # bfloat16 doesn't work in MPS; but it able to learn 18-digit addition on gopher.
    # import torch
    # torch.set_default_dtype(torch.bfloat16)

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct train and test datasets
    train_dataset = AdditionDataset(config.data, split='train')
    test_dataset  = AdditionDataset(config.data, split='test')

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # helper function for the evaluation of a model
    def eval_split(trainer, split, max_batches=None):
        dataset = {'train':train_dataset, 'test':test_dataset}[split]
        ndigit = config.data.ndigit
        results = []
        mistakes_printed_already = 0
        # for larget digit learning, factors will go beyond long64 range, and we need to python magic to deal wih it in cpu
        factors = numpy.array([[10**i for i in range(ndigit + 1)][::-1]])
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
        for b, (x, y) in enumerate(loader):
            x = x.to(trainer.device)
            # isolate the first two digits of the input sequence alone
            d1d2 = x[:, :ndigit*2]
            # let the model sample the rest of the sequence
            d1d2d3 = model.generate(d1d2, ndigit+1, do_sample=False) # using greedy argmax, not sampling
            # isolate the last digit of the sampled sequence
            d3 = d1d2d3[:, -(ndigit+1):]
            d3 = d3.flip(1) # reverse the digits to their "normal" order
            # decode the integers from individual digits
            d1d2 = d1d2.cpu().numpy()
            d3 = d3.cpu().numpy()
            d1i = (d1d2[:,:ndigit] * factors[:,1:]).sum(1)
            d2i = (d1d2[:,ndigit:ndigit*2] * factors[:,1:]).sum(1)
            d3i_pred = (d3 * factors).sum(1)
            d3i_gt = d1i + d2i # manually calculate the ground truth
            # evaluate the correctness of the results in this batch
            correct = (d3i_pred == d3i_gt)
            for i in range(x.size(0)):
                results.append(int(correct[i]))
                if not correct[i] and mistakes_printed_already < 2: # only print up to 5 mistakes to get a sense
                    mistakes_printed_already += 1
                    print("GPT claims that %d + %d = %d but gt is %d" % (d1i[i], d2i[i], d3i_pred[i], d3i_gt[i]))
            if max_batches is not None and b+1 >= max_batches:
                break
        rt = torch.tensor(results, dtype=torch.float)
        print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))
        return rt.sum()

    # iteration callback
    top_score = 0
    losses = []
    def batch_end_callback(trainer):
        global top_score

        true_loss = trainer.loss.item()
        LOSS_AVG_LEN = 5000
        if len(losses) < LOSS_AVG_LEN:
            losses.append(true_loss)
        else:
            losses[trainer.iter_num % LOSS_AVG_LEN] = true_loss
        avg_loss = sum(losses) / len(losses)
        if true_loss < 1e9: # float16 would mess up loss be NaN sometimes?
            gauge('llm.adder.loss', int(true_loss * 1000), tags=["n:" + str(config.data.ndigit), "fixed:1", "model:" + config.model.model_type])
        gauge('llm.adder.iter_time', int(trainer.iter_dt * 1000), tags=["n:" + str(config.data.ndigit), "fixed:1", "model:" + config.model.model_type])

        target_loss = math.sqrt(config.data.ndigit) * 0.01
        if avg_loss < target_loss:
            print(f"Avg loss is {avg_loss}, less than {target_loss}, stopping")
            sys.exit(0)

        if trainer.iter_num % 100 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}; avg loss {avg_loss:.5f}")

        if trainer.iter_num % 10000 == 0:
            # evaluate both the train and test score
            # train_max_batches = {1: None, 2: None, 3: 5}[config.data.ndigit] # if ndigit=2 we can afford the whole train set, ow no
            train_max_batches = 5
            model.eval()
            with torch.no_grad():
                train_score = eval_split(trainer, 'train', max_batches=train_max_batches)
                test_score  = eval_split(trainer, 'test',  max_batches=None)
            score = train_score + test_score
            # save the model if this is the best score we've seen so far
            if score > top_score:
                top_score = score
                print(f"saving model with new top score of {score}")
                ckpt_path = os.path.join(config.system.work_dir, "model.pt")
                torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
