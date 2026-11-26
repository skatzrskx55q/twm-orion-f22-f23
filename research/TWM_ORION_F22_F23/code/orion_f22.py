
"""TWM ORION F22 candidate neural system identification.
Teacher imported only in dataset generation and evaluator, never in model.
"""
import os,copy,json,hashlib,time,random
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.nn import functional as F
import orion_f21 as env
torch.set_num_threads(1)
E,N,K=env.E,env.N,env.K
INC=torch.tensor(env.INC)
EI=torch.tensor(env.EI)
EJ=torch.tensor(env.EJ)

def tensorize(rows):
    return {k:torch.from_numpy(np.stack([r[k] for r in rows])).float() for k in ('node','rel','adj','action','support','target_node','target_rel','actions','future_node','future_rel')}

def samples(seed,split,n=600,steps=4,expose=False):
    bank={'train':env.TRAIN_LAWS,'dev':env.DEV_LAWS,'exam':env.EXAM_LAWS}[split]
    graphs={'train':env.P['graph_train'],'dev':env.P['graph_dev'],'exam':env.P['graph_exam']}[split]
    rng=np.random.default_rng({'train':18100000,'dev':28200000,'exam':38300000}[split]+seed*17777)
    rows=[]
    for i in range(n):
        graph=graphs[i%len(graphs)];adj=env.graph_mask(graph)
        node,rel=env.random_state(rng,adj)
        pre=int(rng.choice([0,1,3,8,20,45,70]))
        start_law=bank[int(rng.integers(len(bank)))]
        for t in range(pre):
            a=env.random_action(rng,adj)
            node,rel=env.teacher_step(node,rel,adj,a,start_law)
        act=env.random_action(rng,adj)
        law=bank[int(rng.integers(len(bank)))]
        sp=env.make_context(rng,adj,[law])[law]
        tn,tr=env.teacher_step(node,rel,adj,act,law)
        actions=np.stack([act]+[env.random_action(rng,adj) for _ in range(steps-1)])
        states=[node];rels=[rel];cn=node.copy();cr=rel.copy()
        for a in actions:
            cn,cr=env.teacher_step(cn,cr,adj,a,law)
            states.append(cn);rels.append(cr)
        rows.append(dict(node=node,rel=rel,adj=adj,action=act,support=sp,target_node=tn,target_rel=tr,
                         actions=actions,future_node=np.stack(states),future_rel=np.stack(rels), law=law, graph=graph))
    tensor=tensorize(rows)
    if expose:tensor.update({'law':[r['law'] for r in rows],'graph':[r['graph'] for r in rows]})
    return tensor

def choose(data,inds):
    return {k:v[inds] for k,v in data.items() if isinstance(v,torch.Tensor)}

class HyperFlow(nn.Module):
    """Shared learned support encoder predicts abstract edge-response parameters.
    Generic tanh activation with full learned coefficients, no encoded law names or teacher.
    """
    def __init__(self,hidden=64,units=1):
        super().__init__()
        self.mem=nn.Sequential(nn.Linear(6,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU())
        self.readout=nn.Sequential(nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,9*units))
        self.scale=nn.Parameter(torch.tensor([.07,.06,.04]))
        self.units=units
        self.inc=nn.Parameter(INC,requires_grad=False)
        self.ei=nn.Parameter(EI,requires_grad=False)
        self.ej=nn.Parameter(EJ,requires_grad=False)
    def latent(self,sup,context=True):
        # direct causal observations; target outputs normalized only by observed action magnitude
        x=torch.stack([sup[...,0]-sup[...,2],sup[...,1]-sup[...,3],sup[...,4]-.5],-1)
        ys=torch.stack([sup[...,7],sup[...,9],sup[...,10]],-1)/sup[...,5:6].clamp_min(.1)
        feat=torch.cat((x,ys/.07),-1)
        if not context:feat=torch.zeros_like(feat)
        z=self.mem(feat).mean(1)
        return self.readout(z).reshape(-1,3,3,self.units)
    def flow(self,node,rel,adj,action,coef):
        x=torch.stack([node[:,self.ei,0]-node[:,self.ej,0],
                       node[:,self.ei,1]-node[:,self.ej,1],rel-.5],-1)
        u=torch.einsum("bei,bicu->becu",x,coef)
        # shared generic single-hidden tanh edge units; law-dependent weights learned from observations
        y=torch.tanh(u).mean(-1)*self.scale
        return y*action[...,None]*adj[...,None]
    def advance(self,node,rel,adj,action,coef):
        f=self.flow(node,rel,adj,action,coef)
        next_node=(node+torch.einsum("en,bec->bnc",self.inc,f[...,:2])).clamp(0,1)
        next_rel=(rel+f[...,2]).clamp(0,1)*adj
        return next_node,next_rel
    def forward(self,node,rel,adj,action,sup,context=True):
        coef=self.latent(sup,context)
        return self.advance(node,rel,adj,action,coef)
    def sequence(self,node,rel,adj,actions,sup,context=True):
        coef=self.latent(sup,context)
        ns=[node];rs=[rel]
        for t in range(actions.shape[1]):
            node,rel=self.advance(node,rel,adj,actions[:,t],coef)
            ns.append(node);rs.append(rel)
        return torch.stack(ns,1),torch.stack(rs,1)

class RidgeFlow(nn.Module):
    """Differentiable shared latent features with episodic ridge system identification."""
    def __init__(self,dim=3,lam=.0002):
        super().__init__()
        self.phi=nn.Sequential(nn.Linear(3,32),nn.Tanh(),nn.Linear(32,3))
        nn.init.zeros_(self.phi[-1].weight);nn.init.zeros_(self.phi[-1].bias)
        self.scale=nn.Parameter(torch.tensor(.2))
        self.ridge=nn.Parameter(torch.tensor(float(np.log(np.exp(lam)-1))))
        self.register_buffer('inc',INC)
        self.register_buffer('ei',EI)
        self.register_buffer('ej',EJ)
    def features(self,x):
        return x+self.scale*self.phi(x)
    def latent(self,sup,context=True):
        x=torch.stack([sup[...,0]-sup[...,2],sup[...,1]-sup[...,3],sup[...,4]-.5],-1)
        ys=torch.stack([sup[...,7],sup[...,9],sup[...,10]],-1)/sup[...,5:6].clamp_min(.1)
        if not context:ys=torch.zeros_like(ys)
        phi=self.features(x)
        lam=F.softplus(self.ridge).clamp(1e-7,.1)
        xt=phi.transpose(-1,-2)
        mat=xt@phi + torch.eye(3)[None]*lam
        return torch.linalg.solve(mat,xt@ys)
    def advance(self,node,rel,adj,action,coef):
        x=torch.stack([node[:,self.ei,0]-node[:,self.ej,0],
                       node[:,self.ei,1]-node[:,self.ej,1],rel-.5],-1)
        f=torch.einsum('bei,bic->bec',self.features(x),coef)*action[...,None]*adj[...,None]
        n=(node+torch.einsum('en,bec->bnc',self.inc,f[...,:2])).clamp(0,1)
        r=(rel+f[...,2]).clamp(0,1)*adj
        return n,r
    def forward(self,node,rel,adj,action,sup,context=True):
        return self.advance(node,rel,adj,action,self.latent(sup,context))
    def sequence(self,node,rel,adj,actions,sup,context=True):
        coef=self.latent(sup,context);ns=[node];rs=[rel]
        for t in range(actions.shape[1]):
            node,rel=self.advance(node,rel,adj,actions[:,t],coef);ns.append(node);rs.append(rel)
        return torch.stack(ns,1),torch.stack(rs,1)

def errs(pred,d):
    n,r=pred
    targetn=d['future_node'];targetr=d['future_rel'];adj=d['adj']
    diffn=(n[:,1:]-targetn[:,1:]).abs()
    diffr=(r[:,1:]-targetr[:,1:]).abs()*adj[:,None]
    mask=(torch.maximum(diffn.amax((2,3)),diffr.amax(2))<=.05)
    return {'node_mae':float(diffn.mean()), 'relation_mae':float(diffr.sum()/adj.sum().clamp_min(1)/r.shape[1]*r.shape[0]), 
            'h1':int(mask[:,0].sum()),'h4':int(mask[:,:4].all(1).sum()),'n':len(n),
            'max4':float(torch.maximum(diffn.amax((2,3)),diffr.amax(2)).amax(1).mean())}
def loss_fn(pred,data,weight=1.):
    n,r=pred;adj=data['adj']
    tn=data['future_node'];tr=data['future_rel']
    weights=torch.linspace(1,weight,n.shape[1]-1)[None,:,None,None]
    ln=((n[:,1:]-tn[:,1:]).square()*weights).mean()/.04**2
    wr=weights.squeeze(-1)
    lr=(((r[:,1:]-tr[:,1:]).square()*adj[:,None])*wr).sum()/adj.sum().clamp_min(1)/(r.shape[1]-1)/.025**2
    return ln+lr

def train_once(arm,seed,train_data,dev_data,steps=1400,seq_length=4,lr=.0013,roll_weight=1.2):
    torch.manual_seed(seed+992)
    model=HyperFlow() if arm=="hyper" else RidgeFlow()
    optim=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
    rng=np.random.default_rng(seed+7777)
    history=[];best=(1e9,1e9);best_state=None;best_step=None
    t0=time.time()
    for step in range(1,steps+1):
        ix=rng.integers(0,len(train_data['node']),size=64)
        b=choose(train_data,torch.tensor(ix))
        out=model.sequence(b['node'],b['rel'],b['adj'],b['actions'][:,:seq_length],b['support'])
        loss=loss_fn(out,{**b,'future_node':b['future_node'][:,:seq_length+1],
                           'future_rel':b['future_rel'][:,:seq_length+1]},roll_weight)
        optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3.);optim.step()
        if step in set([100,250,500,800,1100,1400,2000,2800,3800]) or step==steps:
            model.eval()
            with torch.no_grad():
                dout=model.sequence(dev_data['node'],dev_data['rel'],dev_data['adj'],
                                    dev_data['actions'][:,:seq_length],dev_data['support'])
                de=loss_fn(dout,{**dev_data,'future_node':dev_data['future_node'][:,:seq_length+1],
                            'future_rel':dev_data['future_rel'][:,:seq_length+1]},1.)
                met=errs(dout,{**dev_data,'future_node':dev_data['future_node'][:,:seq_length+1],
                            'future_rel':dev_data['future_rel'][:,:seq_length+1]})
            h={'step':step,'dev_loss':float(de),**met}
            history.append(h)
            score=(h['dev_loss'],h['node_mae']+h['relation_mae'])
            if score<best:best=score;best_state=copy.deepcopy(model.state_dict());best_step=step
            model.train()
    model.load_state_dict(best_state);model.eval()
    return model,history,best_step,time.time()-t0

class WarpRidge(RidgeFlow):
    """Meta-learned shared invertible response link plus generic ridge-adapted coefficients.
    No named laws, hand-entered coefficients, mask enumeration or teacher inference.
    """
    def __init__(self):
        super().__init__()
        # Outputs are bounded by learned global magnitude; causal weights identified per episode.
        self.rawamp=nn.Parameter(torch.tensor([-.8,-.95,-1.1]))
        self.ridge=nn.Parameter(torch.tensor(-11.0))
    def amp(self):
        return .01+F.softplus(self.rawamp)*.1
    def latent(self,sup,context=True):
        x=torch.stack([sup[...,0]-sup[...,2],sup[...,1]-sup[...,3],sup[...,4]-.5],-1)
        ys=torch.stack([sup[...,7],sup[...,9],sup[...,10]],-1)/sup[...,5:6].clamp_min(.1)
        if not context:ys=torch.zeros_like(ys)
        amp=self.amp()
        transformed=torch.atanh((ys/amp).clamp(-.985,.985))
        phi=self.features(x)
        xt=phi.transpose(-1,-2)
        lam=F.softplus(self.ridge).clamp(1e-9,.05)
        return torch.linalg.solve(xt@phi+torch.eye(3)[None]*lam,xt@transformed)
    def advance(self,node,rel,adj,action,coef):
        x=torch.stack([node[:,self.ei,0]-node[:,self.ej,0],
                       node[:,self.ei,1]-node[:,self.ej,1],rel-.5],-1)
        y=self.amp()*torch.tanh(torch.einsum('bei,bic->bec',self.features(x),coef))
        f=y*action[...,None]*adj[...,None]
        n=(node+torch.einsum('en,bec->bnc',self.inc,f[...,:2])).clamp(0,1)
        r=(rel+f[...,2]).clamp(0,1)*adj
        return n,r

class LearnableOddLink(RidgeFlow):
    """Generic trainable monotone polynomial link; per-episode coefficients fitted
    only from observed state changes. No analytic tanh inverse or known teacher law.
    """
    def __init__(self,degree=9):
        super().__init__()
        self.rawamp=nn.Parameter(torch.tensor([-.72,-.85,-1.0]))
        self.coeffs=nn.Parameter(torch.full((3,(degree-1)//2),-2.0))
        self.ridge=nn.Parameter(torch.tensor(-10.))
    def amp(self):
        return .01 + .1*F.softplus(self.rawamp)
    def odd(self,q):
        # monotone odd map with learned nonnegative polynomial coefficients
        c=F.softplus(self.coeffs)
        ans=q
        qpow=q*q
        factor=qpow
        for k in range(c.shape[1]):
            ans=ans+c[:,k]*q*factor
            factor=factor*qpow
        return ans
    def derivative(self,q):
        c=F.softplus(self.coeffs)
        d=torch.ones_like(q)
        qpow=q*q
        factor=qpow
        for k in range(c.shape[1]):
            d=d+(3+2*k)*c[:,k]*factor
            factor=factor*qpow
        return d
    def latent(self,sup,context=True):
        x=torch.stack([sup[...,0]-sup[...,2],sup[...,1]-sup[...,3],sup[...,4]-.5],-1)
        y=torch.stack([sup[...,7],sup[...,9],sup[...,10]],-1)/sup[...,5:6].clamp_min(.1)
        if not context:y=torch.zeros_like(y)
        q=(y/self.amp()).clamp(-.95,.95)
        v=self.odd(q)
        phi=self.features(x);xt=phi.transpose(-1,-2)
        lam=F.softplus(self.ridge).clamp(1e-9,.1)
        return torch.linalg.solve(xt@phi+torch.eye(3)[None]*lam,xt@v)
    def advance(self,node,rel,adj,action,coef):
        x=torch.stack([node[:,self.ei,0]-node[:,self.ej,0],
                       node[:,self.ei,1]-node[:,self.ej,1],rel-.5],-1)
        u=torch.einsum('bei,bic->bec',self.features(x),coef)
        q=u.clamp(-.97,.97)
        for t in range(9):
            q=(q-(self.odd(q)-u)/self.derivative(q)).clamp(-.99,.99)
        y=self.amp()*q
        f=y*action[...,None]*adj[...,None]
        n=(node+torch.einsum('en,bec->bnc',self.inc,f[...,:2])).clamp(0,1)
        r=(rel+f[...,2]).clamp(0,1)*adj
        return n,r
