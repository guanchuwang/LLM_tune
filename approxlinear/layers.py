import torch
import torch.nn.functional as F
from torch import device, topk
from torch.autograd.function import Function
from torch.cuda.amp import custom_fwd, custom_bwd
from .functional import approxlinear_only_bw, approxmatmul_4D_only_bw, RMSNorm_forward, RMSNorm_func

# TASK_TO_NUM_INSTANCE = {"mrpc": 3668,
#                         "cola": 8551,
#                         "stsb": 5749,
#                         'sst2': 66349,
#                         "mnli": 392702,
#                         "mnli_mismatched": None,
#                         "mnli_matched": None,
#                         "qnli": 103743,
#                         "rte": 2490,
#                         "wnli": None,
#                         "qqp": 362846,
#                         "superglue-boolq": None,
#                         "superglue-rte":   None,
#                         "superglue-cb":    None,
#                         "superglue-copa":  None,
#                         "superglue-multirc": None,
#                         "superglue-wic":     None,
#                         "superglue-wsc.fixed": None,
#                         "superglue-record":    None
#                        }


class Scheme(object):

    def __init__(self, lfc_block, hfc_bit):
        self.lfc_block = lfc_block
        self.hfc_bit_num = hfc_bit
        self.max_thread = 1024


class ApproxLinear(torch.nn.Linear):
    def __init__(self, input_features, output_features, bias=True, config=None, batch_dim_use_same_indices=False):
        super(ApproxLinear, self).__init__(input_features, output_features, bias)
        self.batch_dim_use_same_indices = batch_dim_use_same_indices
        self.func = approxlinear_only_bw.apply

        # assert len(config.tasks) == 1 and TASK_TO_NUM_INSTANCE[config.tasks[0]] is not None
        # Scheme.num_samples = TASK_TO_NUM_INSTANCE[config.tasks[0]]

        self.scheme = Scheme(config.lfc_block, config.hfc_bit)

    def forward(self, input):

        if self.training:
             return self.func(input, self.weight, self.bias, self.scheme)
        else:
            return super(ApproxLinear, self).forward(input)


class Approxmatmul_4D(torch.nn.Module):
    def __init__(self, config=None, batch_dim_use_same_indices=True):
        super(Approxmatmul_4D, self).__init__()
        self.batch_dim_use_same_indices = batch_dim_use_same_indices
        self.func = approxmatmul_4D_only_bw.apply
        # self.func = approxmatmul_4D_fw_and_bw.apply

        # assert len(config.tasks) == 1 and TASK_TO_NUM_INSTANCE[config.tasks[0]] is not None
        # Scheme.num_samples = TASK_TO_NUM_INSTANCE[config.tasks[0]]
        self.scheme = Scheme(config.lfc_block, config.hfc_bit)

    def forward(self, A, B):

        if self.training:
            return self.func(A, B, self.scheme)
        else:
            return torch.matmul(A, B)

    # def __str__(self):
    #     return "Approxmatmul_4D"


class ApproxRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6, config=None):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.func = RMSNorm_func
        self.scheme = Scheme(config.lfc_block, config.hfc_bit)

    def forward(self, hidden_states):

        if self.training:
            return self.func(hidden_states, self.weight, self.variance_epsilon, self.scheme)
        else:
            return RMSNorm_forward(self.weight, self.variance_epsilon, hidden_states)


