"""Experimental observed-video KV prefill for the pinned Hand2World AR engine.
Only prefix latents are forced. From the cutoff onward original AR denoising runs.
Original source is guarded and patched in memory, not edited on disk.
"""
import inspect,textwrap

def install(engine_module,context_video,context_frames):
    if (context_frames-1)%4:raise ValueError('History must have 4k+1 frames')
    cls=engine_module.CausalInferenceEngine
    original=cls._prepare_inputs
    def prepare(self):
        result=original(self)
        pixels=engine_module.load_video_frames(context_video,self._resolved_h,self._resolved_w,context_frames,self.cfg.target_fps)
        if self._tae_encode:raise ValueError('Context adapter requires original Wan VAE encoding')
        self._observed_context=engine_module.encode_video_with_vae(self.vae,pixels,self.device,self.dtype)
        assert self._observed_context.shape[2]==1+(context_frames-1)//4
        return result
    source=textwrap.dedent(inspect.getsource(cls.run))
    needle='''        # Block 0: ref short-circuit — use ref frame 0 as clean start; cache-update at t=0.'''
    if source.count(needle)!=1:raise RuntimeError('Upstream AR engine changed; review context hook')
    patch='''        # Observed prefix enters as clean latents, exactly like native block zero.
        if block_idx < self._observed_context.shape[2]:
            x_block = self._observed_context[:, :, sl].clone()
            update_causal_window(self.transformer, cs, ce)
            _fwd(x_block, zeros_t)
            output_latent[:, :, sl] = x_block
            continue

'''
    source=source.replace(needle,patch+needle)
    # Apply first-generated DMD schedule at the first target block after prefix.
    source=source.replace('use_fc = block_idx == 1 and','use_fc = block_idx == self._observed_context.shape[2] and')
    scope=dict(vars(engine_module));exec(compile(source,'<hand_context_prefill>','exec'),scope)
    cls._prepare_inputs=prepare;cls.run=scope['run']
