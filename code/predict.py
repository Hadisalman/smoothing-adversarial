""" This script loads a base classifier and then runs PREDICT on many examples from a dataset.
"""
import argparse
import copy
import datetime
import os
from time import time

from architectures import get_architecture
from attacks import Attacker, PGD_L2, DDN 
from core import Smooth
from datasets import get_dataset, DATASETS, get_num_classes
import setGPU
import torch
from torchvision.transforms import ToPILImage
from train_utils import requires_grad_


parser = argparse.ArgumentParser(description='Predict on many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outdir", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")

#####################
# Attack params
parser.add_argument('--attack', default=None, type=str, choices=['DDN', 'PGD'])
parser.add_argument('--epsilon', default=64.0, type=float)
parser.add_argument('--num-steps', default=100, type=int)
parser.add_argument('--warmup', default=1, type=int)
parser.add_argument('--num-noise-vec', default=1, type=int,
                    help="number of noise vectors to use for finding adversarial examples")
parser.add_argument('--no-grad-attack', action='store_true',
                    help="Choice of whether to use gradients during attack or do the cheap trick")
parser.add_argument('--base-attack', action='store_true')
parser.add_argument("--visualize-examples", action='store_true', help="Whether to save the adversarial examples or not")

# PGD-specific
parser.add_argument('--random-start', default=True, type=bool)

# DDN-specific
parser.add_argument('--init-norm-DDN', default=256.0, type=float)
parser.add_argument('--gamma-DDN', default=0.05, type=float)


args = parser.parse_args()

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

toPilImage = ToPILImage()

if __name__ == "__main__":

    if args.attack:
        if args.base_attack:
            outdir = '{}/{}_{}_base'.format(args.outdir, args.attack, args.epsilon, args.sigma)
        else:
            outdir = '{}/{}_{}_noise_{:.3}_mtest_{}'.format(args.outdir, args.attack, args.epsilon, args.sigma, args.num_noise_vec)
    
    else:
        outdir = '{}/noise_{:.3}'.format(args.outdir, args.sigma)

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if args.visualize_examples:
        adv_examples_dir = '{}/adv_examples'.format(outdir)
        if not os.path.exists(adv_examples_dir):
            os.makedirs(adv_examples_dir)
    outfile = os.path.join(outdir, 'predictions')

    args.epsilon /= 256.0
    args.init_norm_DDN /= 256.0
    if args.epsilon > 0:
        args.gamma_DDN = 1 - (3/510/args.epsilon)**(1/args.num_steps)

    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    base_classifier.eval()
    requires_grad_(base_classifier, False)
    
    # create the smoothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    f = open(outfile, 'w')
    print("idx\tlabel\tpredict\tbasePredict\tcorrect\ttime", file=f, flush=True)


    if args.attack == 'PGD':
        print('Attacker is PGD')
        attacker = PGD_L2(steps=args.num_steps, device='cuda', max_norm=args.epsilon)
        # attacker = PGD(num_steps=args.num_steps, norm='l2', step_size=args.epsilon/args.num_steps*2, epsilon=args.epsilon)
    elif args.attack == 'DDN':
        print('Attacker is DDN')
        attacker = DDN(steps=args.num_steps, device='cuda', max_norm=args.epsilon, 
                    init_norm=args.init_norm_DDN, gamma=args.gamma_DDN)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    base_smoothed_agree = 0
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        before_time = time()

        (x, label) = dataset[i]

        if args.visualize_examples and i < 1000:
            pil = toPilImage(x)
            pil.save("{}/{}_clean.png".format(adv_examples_dir, i))

        x = x.unsqueeze(0).cuda()
        label = torch.Tensor([label]).cuda().long()

        if args.attack in ['PGD', 'DDN']:
            x = x.repeat((args.num_noise_vec, 1, 1, 1))

            noise = (1 - int(args.base_attack))* torch.randn_like(x, device='cuda') * args.sigma
            x = attacker.attack(base_classifier, x, label, 
                                noise=noise, num_noise_vectors=args.num_noise_vec,
                                no_grad=args.no_grad_attack,
                                )
            x = x[:1]

            if args.visualize_examples and i < 1000:
                pil = toPilImage(x.detach()[0].cpu())
                pil.save("{}/{}_adv.png".format(adv_examples_dir, i))
            

        base_output = base_classifier(x)
        base_prediction = base_output.argmax(1).item()
        
        x = x.squeeze()
        label = label.item()
        # make the prediction
        prediction = smoothed_classifier.predict(x, args.N, args.alpha, args.batch)
        if base_prediction == prediction:
            base_smoothed_agree += 1 
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

        # log the prediction and whether it was correct
        print("{}\t{}\t{}\t{}\t{}\t{}".format(i, label, prediction, base_prediction, correct, time_elapsed), file=f, flush=True)
    f.close()

