import torch

from lib.train.distributed import is_main_process


def _newton_schulz_impl(G, steps=5):
    """Batched Newton-Schulz orthogonalization (supports ndim >= 2)"""

    a, b, c = (3.4445, -4.7750, 2.0315)
    X       = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


_newton_schulz = torch.compile(_newton_schulz_impl, dynamic=None)


class MuonE2E:
    """End-to-end Muon optimizer for ALL parameter types

    ndim >= 2: Newton-Schulz ortho + aspect ratio scaling (includes embeddings)
    ndim == 1: L2-normalized momentum
    """

    def __init__(self, model, lr, weight_decay, momentum=0.95):

        self._params_2d = []
        self._params_1d = []

        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2:
                self._params_2d.append(p)
            else:
                self._params_1d.append(p)

        self._bufs_2d = [torch.zeros_like(p) for p in self._params_2d]
        self._bufs_1d = [torch.zeros_like(p) for p in self._params_1d]
        self._momentum = momentum
        self._wd       = weight_decay

        self._lr_proxy   = torch.optim.SGD([torch.zeros(1)], lr=lr)
        self._schedulers = []

    def step(self):

        lr = self._lr_proxy.param_groups[0]['lr']

        active    = [i for i, p in enumerate(self._params_2d) if p.grad is not None]
        grads_2d  = [self._params_2d[i].grad for i in active]
        params_2d = [self._params_2d[i].data for i in active]
        bufs_2d   = [self._bufs_2d[i] for i in active]

        torch._foreach_lerp_(bufs_2d, grads_2d, 1 - self._momentum)
        torch._foreach_lerp_(grads_2d, bufs_2d, self._momentum)

        for j, i in enumerate(active):
            p         = self._params_2d[i]
            update    = _newton_schulz(grads_2d[j])
            update   *= max(1, p.size(-2) / p.size(-1)) ** 0.5
            grads_2d[j] = update

        torch._foreach_mul_(params_2d, 1 - lr * self._wd)
        torch._foreach_add_(params_2d, grads_2d, alpha=-lr)

        active_1d = [i for i, p in enumerate(self._params_1d) if p.grad is not None]
        if active_1d:
            grads_1d  = [self._params_1d[i].grad for i in active_1d]
            params_1d = [self._params_1d[i].data for i in active_1d]
            bufs_1d   = [self._bufs_1d[i] for i in active_1d]

            torch._foreach_lerp_(bufs_1d, grads_1d, 1 - self._momentum)
            torch._foreach_lerp_(grads_1d, bufs_1d, self._momentum)

            norms = torch._foreach_norm(grads_1d)
            torch._foreach_add_(norms, 1e-7)
            torch._foreach_div_(grads_1d, norms)

            torch._foreach_mul_(params_1d, 1 - lr * self._wd)
            torch._foreach_add_(params_1d, grads_1d, alpha=-lr)

    def zero_grad(self, set_to_none=False):

        for p in self._params_2d:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

        for p in self._params_1d:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def state_dict(self):

        return {
            'muon_e2e': {
                'bufs_2d':    [buf.clone() for buf in self._bufs_2d],
                'bufs_1d':    [buf.clone() for buf in self._bufs_1d],
                'lr_proxy':   self._lr_proxy.state_dict(),
                'schedulers': [s.state_dict() for s in self._schedulers],
            }
        }

    def load_state_dict(self, state_dict):
        """Load state; silently skip incompatible checkpoints (AdamW, MuonAdamW)"""

        if 'muon_e2e' not in state_dict:
            return
        s = state_dict['muon_e2e']
        if 'bufs_2d' in s:
            for buf, saved in zip(self._bufs_2d, s['bufs_2d']):
                buf.copy_(saved)
        if 'bufs_1d' in s:
            for buf, saved in zip(self._bufs_1d, s['bufs_1d']):
                buf.copy_(saved)
        if 'lr_proxy' in s:
            self._lr_proxy.load_state_dict(s['lr_proxy'])
        if 'schedulers' in s and self._schedulers:
            for sched, saved in zip(self._schedulers, s['schedulers']):
                sched.load_state_dict(saved)

    def fast_forward_schedulers(self, target_step):
        """Advance schedulers to target_step (for old checkpoints without scheduler state)"""

        for s in self._schedulers:
            for _ in range(target_step - s.last_epoch):
                s.step()

    def create_schedulers(self, schedule_fn, warmup_steps, total_steps):

        self._schedulers = [schedule_fn(self._lr_proxy, warmup_steps, total_steps)]

    def step_schedulers(self):

        for s in self._schedulers:
            s.step()

    def get_last_lr(self):

        return self._schedulers[0].get_last_lr() if self._schedulers else [0.0]

    def parameters(self):

        yield from self._params_2d
        yield from self._params_1d

    @property
    def param_groups(self):

        return self._lr_proxy.param_groups


def create_optimizer(model, lr, weight_decay):
    """Create Muon E2E optimizer"""

    opt = MuonE2E(model, lr, weight_decay)
    if is_main_process():
        n_2d = sum(p.numel() for p in opt._params_2d)
        n_1d = sum(p.numel() for p in opt._params_1d)
        print(f"  MuonE2E: {n_2d + n_1d:,} params (lr={lr})")
        print(f"    2D/3D: {len(opt._params_2d)} tensors, {n_2d:,} params (Newton-Schulz)")
        print(f"    1D:    {len(opt._params_1d)} tensors, {n_1d:,} params (L2-norm)")
    return opt
