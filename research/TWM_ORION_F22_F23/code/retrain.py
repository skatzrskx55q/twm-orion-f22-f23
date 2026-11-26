
#!/usr/bin/env python3
"""Re-train F22 learned monotone response model on TRAIN/DEV (not on test).
Training seeds are now explicitly initialized for reproducible new replicates;
archived F22 first-seed initialization state was not retained, so bitwise
identity with provided checkpoint is NOT promised.
"""
import argparse, copy, json, sys, time, zipfile
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]

def main(seed,steps):
    from reproduce import import_deps
    env,p2,f22,f23=import_deps()
    torch.set_num_threads(1)
    if seed not in (1701,1703):raise ValueError('Prespecified seed 1701 or 1703')
    train=f23.samples(seed,'train',n=900,steps=4)
    dev=f23.samples(seed,'dev',n=144,steps=4)
    # Explicit seed for NEW retraining replicate; archived original RNG not fully logged.
    torch.manual_seed(seed+2026)
    net=f23.LearnableOddLink(degree=11)
    optim=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=1e-6)
    rng=np.random.default_rng(seed+5009)
    best=float('inf');state=None;best_step=None;hist=[]
    for step in range(1,steps+1):
        b=f23.choose(train,torch.tensor(rng.integers(0,len(train['node']),size=64)))
        pred=net.sequence(b['node'],b['rel'],b['adj'],b['actions'][:,:4],b['support'])
        loss=f23.loss_fn(pred,b,1.2)
        optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2)
        optim.step()
        if step in (100,250,500,800,1200,1600,1800):
            net.eval()
            with torch.no_grad():
                out=net.sequence(dev['node'],dev['rel'],dev['adj'],dev['actions'][:,:4],dev['support'])
                val=float(f23.loss_fn(out,dev,1.0))
                met=f23.errs(out,dev)
            hist.append({'step':step,'dev_loss':val,**met})
            if val<best:best=val;state=copy.deepcopy(net.state_dict());best_step=step
            net.train()
    if state is None:raise RuntimeError('No checkpoint evaluation step reached')
    outdir=ROOT/'retraining_outputs'
    outdir.mkdir(exist_ok=True)
    torch.save({'seed':seed,'model':state,'dev_step':best_step},outdir/f'RETRAINED_{seed}.pt')
    (outdir/f'RETRAINED_{seed}.json').write_text(json.dumps({
        'seed':seed,'best_step':best_step,'train_seed_explicit':seed+2026,'history':hist},indent=2))
    print(seed,'best step',best_step,'dev loss',best)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--steps',type=int,default=1800)
    args=ap.parse_args()
    main(args.seed,args.steps)
