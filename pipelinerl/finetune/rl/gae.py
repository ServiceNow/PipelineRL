import torch

def compute_gae_advantages(
    rewards: torch.Tensor,          # [B, T] (or [T])
    value_pred: torch.Tensor,       # [B, T] (or [T])
    lamda,                          # float or Tensor: [B], [B,T], or scalar
    gamma: float = 1.0,
    mask: torch.Tensor | None = None,      # [B, T] bool, valid tokens (highly recommended)
    segments=None,                        # optional: list of (b, start, end) or (start, end)
    logger = None,
):
    
    with torch.no_grad():
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(0)
            value_pred = value_pred.unsqueeze(0)
            if mask is not None and mask.ndim == 1:
                mask = mask.unsqueeze(0)

        B, T = rewards.shape
        device = rewards.device
        dtype = rewards.dtype

        # Build continuation mask cont[:, t] meaning "t continues to t+1 in same episode"
        # Default: all positions except last continue
        cont = torch.ones((B, T), device=device, dtype=dtype)
        cont[:, -1] = 0.0

        if mask is not None:
            # Only allow continuation if both current and next tokens are valid
            m = mask.to(device=device, dtype=dtype)
            cont = cont * m
            cont[:, :-1] = cont[:, :-1] * m[:, 1:]
            cont[:, -1] = 0.0

        # If packed segments are provided, force terminals at each segment end
        # Expected tuples:
        #  - (b, start, end) where end is the last token index in that segment (inclusive)
        #  - or (start, end) if you have a single-row packed layout
        if segments is not None:
            for seg in segments:
                if len(seg) == 3:
                    b, start, end = seg
                elif len(seg) == 2:
                    start, end = seg
                    b = 0
                else:
                    raise ValueError(f"Bad segment tuple: {seg}")

                if 0 <= b < B and 0 <= end < T:
                    cont[b, end] = 0.0

        # Broadcast lamda to [B, T]
        if not isinstance(lamda, torch.Tensor):
            lam = torch.full((B, T), float(lamda), device=device, dtype=dtype)
        else:
            lam = lamda.to(device=device, dtype=dtype)
            if lam.ndim == 0:
                lam = lam.expand(B, T)
            elif lam.ndim == 1:
                # [B] -> [B, T]
                lam = lam[:, None].expand(B, T)
            elif lam.ndim == 2:
                # [B, T]
                pass
            else:
                raise ValueError(f"lamda must be scalar/[B]/[B,T], got shape {tuple(lam.shape)}")

        # GAE should expect reward to be 1 at the last token and 0 elsewhere
        # currently the entire sequence gets a reward of 1 if the answer was correct, we'll fix this here
        # terminal is 1 at end-of-segment tokens, 0 otherwise
        terminal = (1.0 - cont)
        print(terminal)

        # (optional but recommended) don't ever put reward on invalid/padded tokens
        if mask is not None:
            terminal = terminal * mask.to(device=device, dtype=dtype)

        # keep reward only at terminal tokens
        rewards = rewards * terminal

        # logger.info("REWARD_DEBUG")
        # logger.info(segments)
        # logger.info(rewards)
        # logger.info(rewards.sum())


        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros((B,), device=device, dtype=dtype)

        for t in range(T - 1, -1, -1):
            if t == T - 1:
                next_v = torch.zeros((B,), device=device, dtype=dtype)
            else:
                next_v = value_pred[:, t + 1]

            c = cont[:, t]  # 0 if terminal at t, else 1
            # breakpoint()
            # logger.info("======DEBUG======")
            # logger.info(t)
            # logger.info(gamma)
            # logger.info(rewards)
            # logger.info(next_v)
            # logger.info(c)
            # logger.info(value_pred)
            # logger.info(lam[:t])
            # logger.info("==================")
            delta = rewards[:, t] + gamma * next_v * c - value_pred[:, t]
            # logger.info(delta)
            last_gae = delta + gamma * lam[:, t] * c * last_gae
            advantages[:, t] = last_gae

        returns = advantages + value_pred
        return advantages, returns

if __name__ == "__main__":
    rewards = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    # values  = torch.tensor([[0.1, 0.3, 0.8, 0.7, 0.8, 0.1, 0.3, 0.8, 0.7, 0.8]])
    values  = torch.zeros_like(rewards)
    segments = [(0, 4), (4, 9)]

    advantages, returns = compute_gae_advantages(
        rewards=rewards,
        value_pred=values,
        lamda=0.95,
    )

    print("advantages:", advantages)

    # rewards = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    # values  = torch.zeros_like(rewards)

    # lamda = torch.tensor([[0.0, 0.0, 1.0, 1.0]])  # early stop, then accumulate

    # advantages, _ = compute_gae_advantages(
    #     rewards=rewards,
    #     value_pred=values,
    #     lamda=lamda,
    #     gamma=1.0,
    # )

    # print("advantages:", advantages)


    # rewards = torch.tensor([[1.0, 1.0, 10.0, 10.0]])
    # values  = torch.zeros_like(rewards)

    # # End segment at index 1
    # segments = [(0, 0, 1), (0, 2, 3)]

    # advantages, _ = compute_gae_advantages(
    #     rewards=rewards,
    #     value_pred=values,
    #     lamda=1.0,
    #     gamma=1.0,
    #     segments=segments,
    # )

    # print("advantages:", advantages)
    # rewards = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    # values  = torch.zeros_like(rewards)
    # mask    = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)

    # advantages, _ = compute_gae_advantages(
    #     rewards=rewards,
    #     value_pred=values,
    #     lamda=1.0,
    #     gamma=1.0,
    #     mask=mask,
    # )

    # print("advantages:", advantages)

    # rewards = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
    # values  = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]])

    # advantages, _ = compute_gae_advantages(
    #     rewards=rewards,
    #     value_pred=values,
    #     lamda=0.99,
    #     gamma=1.0,
    # )

    # print("advantages:", advantages)


