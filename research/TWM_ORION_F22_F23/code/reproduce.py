
#!/usr/bin/env python3
"""Recompute saved TWM ORION F22/F23 independent tests from frozen checkpoints.
All teacher procedures in this file belong exclusively to offline evaluation.
No teacher procedure is called by the neural forward.
"""
import argparse, json, sys, zipfile
from pathlib import Path
import numpy as np
import torch
from scipy.special import erf

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)

def import_deps():
    deps=ROOT/'dependency'
    old=deps/'TWM_ORION_F21'/'code'/'orion_f21.py'
    if not old.is_file():
        zpath=ROOT/'upstream_F21.zip'
        with zipfile.ZipFile(zpath) as z:
            for member in z.namelist():
                p=Path(member)
                if p.is_absolute() or '..' in p.parts:raise RuntimeError("Unsafe ZIP member")
            deps.mkdir(exist_ok=True)
            z.extractall(deps)
    for folder in (old.parent,ROOT/'code'):
        if str(folder) not in sys.path:sys.path.insert(0,str(folder))
    import orion_f21 as p0
    import orion_f21_p2 as p2
    import orion_f22 as f22
    import orion_f23 as f23
    return p0,p2,f22,f23

def as_tensor(x):
    return torch.tensor(np.asarray(x,dtype=np.float32))
def load_model(kind,seed,f23):
    if kind=='F22':
        net=f23.LearnableOddLink(degree=11)
        fname=ROOT/'checkpoints'/f'oddlink_{seed}.pt'
    elif kind=='F23':
        net=f23.AdaptiveOddLink(degree=7)
        fname=ROOT/'checkpoints'/f'adaptive_{seed}.pt'
    else:raise ValueError(kind)
    net.load_state_dict(torch.load(fname,map_location='cpu',weights_only=True))
    return net.eval()

def exam(seed,p0,p2,f22,f23):
    cand=load_model('F22',1701 if seed==2701 else 1703,f23)
    old=p2.load(1521 if seed==2701 else 1523,'observed_context')
    data=f22.samples(seed,'exam',n=320,steps=4)
    results={}
    for name,net in [('candidate',cand),('parent_F21_P2',old)]:
        with torch.no_grad():pn,pr=net(data['node'],data['rel'],data['adj'],data['action'],data['support'],True)[:2]
        nodeerr=(pn-data['target_node']).abs()
        relerr=(pr-data['target_rel']).abs()*data['adj']
        maxe=torch.maximum(nodeerr.amax((1,2)),relerr.amax(1))
        results[name]={
          'n':len(maxe),'exact_0.02':int((maxe<=.02).sum()),
          'near_0.05':int((maxe<=.05).sum()),
          'node_mae':float(nodeerr.mean()),'rel_mae':float(relerr.sum()/data['adj'].sum())}
    saved=json.loads((ROOT/'results'/f'EXAM_{seed}.json').read_text())
    for k,v in results.items():
        for metric in ('n','exact_0.02','near_0.05'):
            assert v[metric]==saved[k][metric],(seed,k,metric,v[metric],saved[k][metric])
        for metric in ('node_mae','rel_mae'):
            assert abs(v[metric]-saved[k][metric])<1e-7,(seed,k,metric,v[metric],saved[k][metric])
    return results

def make_roll_episode(seed,group,j,p0,horizon=90):
    seed0=912345671+seed*9173+j*6317+{'ood':0,'familiar':10000000,'scrambled':20000000}[group]
    rng=np.random.default_rng(seed0)
    graphs=p0.P['graph_train'] if group=='familiar' else p0.P['graph_exam']
    laws=p0.TRAIN_LAWS if group=='familiar' else p0.EXAM_LAWS
    graph=graphs[j%len(graphs)];law=laws[j%len(laws)]
    adj=p0.graph_mask(graph);node,rel=p0.random_state(rng,adj)
    sup=p0.make_context(rng,adj,[law])[law]
    actions=[p0.random_action(rng,adj) for _ in range(horizon)]
    if group=='scrambled':actions=[actions[(t*37+11)%horizon] for t in range(horizon)]
    tn=[];tr=[];gn,gr=node.copy(),rel.copy()
    for a in actions:
        gn,gr=p0.teacher_step(gn,gr,adj,a,law)
        tn.append(gn);tr.append(gr)
    return {'group':group,'law':law,'graph':graph,'node':node,'rel':rel,
       'adj':adj,'support':sup,'actions':np.stack(actions),'true_n':np.stack(tn),'true_r':np.stack(tr)}

def score_rollouts(episodes,models):
    val=lambda field:as_tensor(np.stack([e[field] for e in episodes]))
    node=val('node');rel=val('rel');adj=val('adj');sp=val('support');actions=val('actions')
    tn=val('true_n');tr=val('true_r')
    out={}
    for name,net in models.items():
        with torch.no_grad():
            pn=node.clone();pr=rel.clone()
            latent=None if name=='parent_F21_P2' or name=='F21_P2' else net.latent(sp,True)
            errs=[]
            for t in range(actions.shape[1]):
                if name in ('parent_F21_P2','F21_P2'):
                    pn,pr=net(pn,pr,adj,actions[:,t],sp,True)[:2]
                else:pn,pr=net.advance(pn,pr,adj,actions[:,t],latent)
                errs.append(torch.maximum((pn-tn[:,t]).abs().amax((1,2)),((pr-tr[:,t]).abs()*adj).amax(1)))
            er=torch.stack(errs,1)
        r={}
        for tol in (.02,.05):
            for h in (1,4,16,90):r[f'H{h}_{tol}']=int((er[:,:h].amax(1)<=tol).sum())
        r['max_error90']=float(er.max())
        r['H90_per_case']=[int(x) for x in (er.amax(1)<=.05).tolist()]
        out[name]=r
    return out

def rollout(seed,p0,p2,f22,f23):
    episodes=[make_roll_episode(seed,group,j,p0) for group,num in [('ood',16),('familiar',8),('scrambled',8)] for j in range(num)]
    names={'candidate':load_model('F22',1701 if seed==2701 else 1703,f23),
           'parent_F21_P2':p2.load(1521 if seed==2701 else 1523,'observed_context')}
    out=score_rollouts(episodes,names)
    saved=json.loads((ROOT/'results'/f'ROLLOUT_{seed}.json').read_text())
    for key in names:
        for h in [1,4,16,90]:
            for tol in [0.02,0.05]:
                metric=f'H{h}_{tol}'
                assert out[key][metric]==saved[key]['all'][metric],(seed,key,metric)
    return {name:{k:v for k,v in row.items() if k.startswith('H')} for name,row in out.items()}

def novel_teacher(node,rel,adj,act,theta,kind,p0):
    du=node[p0.EI,0]-node[p0.EJ,0]
    dp=node[p0.EI,1]-node[p0.EJ,1]
    rv=rel-.5
    u=[theta[0]*du+theta[1]*dp+theta[2]*rv,
       theta[1]*du-theta[0]*dp+theta[2]*rv,
       theta[2]*du+theta[0]*dp+theta[1]*rv]
    if kind=='arctan_response':fn=lambda x:2/np.pi*np.arctan(np.pi*x/2)
    elif kind=='erf_response':fn=lambda x:erf(x*np.sqrt(np.pi)/2)
    elif kind=='cubic_response':fn=lambda x:np.tanh(x+.3*x**3)
    else:raise ValueError(kind)
    flows=np.stack([.085*fn(u[0]),.055*fn(u[1]),.04*fn(u[2])],-1)*act[:,None]*adj[:,None]
    nn=(np.clip(node+np.einsum('en,ec->nc',p0.INC,flows[:,:2]),0,1)).astype(np.float32)
    rr=(np.clip(rel+flows[:,2],0,1)*adj).astype(np.float32)
    return nn,rr

def novel_support(rng,adj,theta,kind,p0,K=96):
    toks=[]
    for k in range(K):
        node,rel=p0.random_state(rng,adj)
        if k%3==0:node[:,0]=np.linspace(.18,.82,p0.N,dtype=np.float32)
        if k%3==1:node[:,1]=np.linspace(.82,.18,p0.N,dtype=np.float32)
        act=p0.random_action(rng,adj,solo=True)
        after=novel_teacher(node,rel,adj,act,theta,kind,p0)
        ix=int(np.flatnonzero(act)[0])
        toks.append(p0.support_token(node,rel,adj,act,*after,ix))
    return np.stack(toks)

def novel_episode(seed,kind,j,p0,graph_mask_override=None,structural=False):
    if structural:
        rng=np.random.default_rng(331888808+seed*3311+j*9113)
        kind=('arctan_response','erf_response','cubic_response')[(j//4)%3]
        graph=('complete6','alternating_bipartite','wheel6')[(j//12)%3]
        adj=graph_mask_override(graph,p0)
    else:
        offs={'arctan_response':0,'erf_response':10000000,'cubic_response':20000000}[kind]
        rng=np.random.default_rng(701234567+seed*1747+offs+j*3907)
        graph=p0.P['graph_exam'][j%2]
        adj=p0.graph_mask(graph)
    law=p0.EXAM_LAWS[j%2];theta=p0.theta(law)
    node,rel=p0.random_state(rng,adj);sup=novel_support(rng,adj,theta,kind,p0)
    actions=np.stack([p0.random_action(rng,adj) for _ in range(90)])
    gn,gr=node.copy(),rel.copy();truthn=[];truthr=[]
    for a in actions:
        gn,gr=novel_teacher(gn,gr,adj,a,theta,kind,p0)
        truthn.append(gn);truthr.append(gr)
    return {'group':kind,'graph':graph,'law':law,'node':node,'rel':rel,
       'adj':adj,'support':sup,'actions':actions,
       'true_n':np.stack(truthn),'true_r':np.stack(truthr)}

def other_graph_mask(name,p0):
    if name=='complete6':pairs=[(i,j) for i in range(p0.N) for j in range(i+1,p0.N)]
    elif name=='alternating_bipartite':pairs=[(i,j) for i in (0,2,4) for j in (1,3,5)]
    elif name=='wheel6':pairs=[(0,j) for j in range(1,6)]+[(1,2),(2,3),(3,4),(4,5),(1,5)]
    else:raise ValueError(name)
    target={tuple(sorted(p)) for p in pairs}
    return np.array([float(x in target) for x in p0.EDGES],dtype=np.float32)

def transfer(seed,kind,p0,p2,f22,f23):
    episodes=[novel_episode(seed,kind,j,p0) for j in range(16)]
    trainseed=1701 if seed==60101 else 1703
    models={'adaptive_odd':load_model('F23',trainseed,f23),
            'F22_odd':load_model('F22',trainseed,f23),
            'F21_P2':p2.load(1521 if seed==60101 else 1523,'observed_context')}
    out=score_rollouts(episodes,models)
    saved=json.loads((ROOT/'results'/f'F23_FINAL_{kind}_{seed}.json').read_text())
    for name in models:
        assert out[name]['H90_0.05']==saved[name]['metrics']['H90_0.05'],(kind,seed,name)
    return {name:o['H90_0.05'] for name,o in out.items()}

def structural(seed,p0,p2,f22,f23):
    episodes=[novel_episode(seed,None,j,p0,graph_mask_override=other_graph_mask,structural=True) for j in range(36)]
    trainseed=1701 if seed==77101 else 1703
    models={'adaptive':load_model('F23',trainseed,f23),
            'F22_odd':load_model('F22',trainseed,f23),
            'F21_P2':p2.load(1521 if seed==77101 else 1523,'observed_context')}
    out=score_rollouts(episodes,models)
    saved=json.loads((ROOT/'results'/f'F23_STRUCTURAL_OOD_{seed}.json').read_text())
    for name in models:
        assert out[name]['H90_0.05']==saved[name]['H90'],(seed,name)
    return {name:o['H90_0.05'] for name,o in out.items()}

def smoke(p0,p2,f22,f23):
    from torch.nn import functional as F
    net=load_model('F23',1701,f23)
    s=f22.samples(2701,'exam',n=4,steps=4)
    with torch.no_grad():
        out=net(s['node'],s['rel'],s['adj'],s['action'],s['support'])
        assert out[0].shape==(4,p0.N,2) and out[1].shape==(4,p0.E)
        assert torch.isfinite(out[0]).all()
        order=torch.randperm(s['support'].shape[1])
        alt=net(s['node'],s['rel'],s['adj'],s['action'],s['support'][:,order])
        assert torch.allclose(out[0],alt[0],atol=1e-5)
        assert torch.allclose(out[1],alt[1],atol=1e-5)
    assert not set(net.advance.__code__.co_names)&{'teacher_step','new_teacher_step','theta'}
    assert not set(net.latent.__code__.co_names)&{'teacher_step','theta','novel_teacher'}
    return 'PASS: load, shape, finite, support permutation invariant, no teacher forward'

def main(scope):
    p0,p2,f22,f23=import_deps()
    summary={'smoke':smoke(p0,p2,f22,f23)}
    if scope in ('exam','all'):
        summary['exam']={str(s):exam(s,p0,p2,f22,f23) for s in (2701,2703)}
    if scope in ('rollout','all'):
        summary['rollout']={str(s):rollout(s,p0,p2,f22,f23) for s in (2701,2703)}
    if scope in ('transfer','all'):
        summary['transfer']={kind:{str(s):transfer(s,kind,p0,p2,f22,f23) for s in (60101,60103)}
          for kind in ('arctan_response','erf_response','cubic_response')}
        summary['structural']={str(s):structural(s,p0,p2,f22,f23) for s in (77101,77103)}
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--scope',choices=['smoke','exam','rollout','transfer','all'],default='smoke')
    args=ap.parse_args()
    main(args.scope)
