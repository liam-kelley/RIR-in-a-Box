    # def deemphasize_rir_early_reflections(self, estimated_rir_batch : torch.Tensor, label_rir_batch : torch.Tensor, t0 : int = 2000):
    #     '''
    #     DEPRECATED
    #     linear ramp up from 0 to 1 on the interval [0,t0] in samples
    #     '''
    #     device = estimated_rir_batch.device
    #     ramp = torch.arange(0, t0, device=device) / t0
    #     # Determine the actual length to use (the minimum of t0 and the last dimension of the tensor)
    #     actual_length = min(t0, estimated_rir_batch.shape[-1])
    #     # Adjust ramp size if necessary
    #     if actual_length < t0:
    #         ramp = ramp[:actual_length]
    #     # Applying the ramp to both estimated_rir_batch and label_rir_batch up to the actual_length
    #     estimated_rir_batch[..., :actual_length] = estimated_rir_batch[..., :actual_length] * ramp.unsqueeze(0)
    #     label_rir_batch[..., :actual_length] = label_rir_batch[..., :actual_length] * ramp.unsqueeze(0)

    #     return estimated_rir_batch, label_rir_batch