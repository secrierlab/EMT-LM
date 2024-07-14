import attr


@attr.s
class model_para_base:
    ops_class_name:list=["custom_norm","fast_attention"]
    ops_class_para:list=[None,None]