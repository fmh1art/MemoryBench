"""Use PyTorch SDPA for upstream calls that bypass --fa_version=0.
Preserves per-sample lengths, scaling and bottom-right causal/window alignment.
Only patches this inference process; original repository files stay intact.
"""
import runpy
import torch

def sdpa(q,k,v,q_lens=None,k_lens=None,dropout_p=0.,softmax_scale=None,
         q_scale=None,causal=False,window_size=(-1,-1),deterministic=False,
         dtype=torch.bfloat16,version=None):
    original_dtype=q.dtype
    result=torch.zeros((*q.shape[:-1],v.shape[-1]),device=q.device,dtype=original_dtype)
    for i in range(q.shape[0]):
        nq=int(q_lens[i]) if q_lens is not None else q.shape[1]
        nk=int(k_lens[i]) if k_lens is not None else k.shape[1]
        qi=q[i,:nq].transpose(0,1).to(dtype)
        ki=k[i,:nk].transpose(0,1).to(dtype)
        vi=v[i,:nk].transpose(0,1).to(dtype)
        if q_scale is not None:qi=qi*q_scale
        mask=None
        if causal or window_size!=(-1,-1):
            row=torch.arange(nq,device=q.device)[:,None]+nk-nq
            col=torch.arange(nk,device=q.device)[None,:]
            mask=torch.ones((nq,nk),device=q.device,dtype=torch.bool)
            if causal:mask &= col<=row
            if window_size[0]>=0:mask &= col>=row-window_size[0]
            if window_size[1]>=0:mask &= col<=row+window_size[1]
        out=torch.nn.functional.scaled_dot_product_attention(qi.unsqueeze(0),ki.unsqueeze(0),vi.unsqueeze(0),attn_mask=mask,dropout_p=dropout_p,scale=softmax_scale)
        result[i,:nq]=out.squeeze(0).transpose(0,1).to(original_dtype)
    return result

if __name__=='__main__':
    import wan.modules.model as model
    model.flash_attention=sdpa
    runpy.run_path('generate.py',run_name='__main__')
