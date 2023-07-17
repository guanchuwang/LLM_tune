from dataclasses import dataclass


@dataclass
class ApproxLinearConfig(object):
    lfc_block = 16
    hfc_bit = 2

    # sampling_ratio = 1.0
    # minimal_k = 10
    # only_bw = True
    # tasks = ["dummy_task"]
    # deter_ratio      = 0.5
    # deter_adaptive   = True
    # sample_replacement = True
    # mix_replacement = False
    # k_sampling       = True
    # q_sampling       = True
    # v_sampling       = True
    # o_sampling       = True
    # wi_0_sampling    = True
    # wi_1_sampling    = True
    # wo_sampling      = True
    # score_sampling   = True
    # attout_sampling  = True
