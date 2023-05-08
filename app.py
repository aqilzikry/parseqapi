import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True, refine_iters=0).eval()
dummy_input = torch.rand(1, 3, *parseq.hparams.img_size)  # (1, 3, 32, 128) by default

traced_script_module = torch.jit.trace(parseq, dummy_input)
optimized_traced_model = optimize_for_mobile(traced_script_module)
optimized_traced_model._save_for_lite_interpreter("mobile_model.pt")