"""
general imports

modules here basically come from scBERT and scFormer
"""

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False