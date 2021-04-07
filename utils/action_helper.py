"""
Module for declaring misc functions to work with actions
"""
from typing import List

def full_action_2_partial_action(x: List[int],mask: List[int]):
    """
    Function that converts the list of actions x into
    a shorter list given the mask

    Args:
        x(List[int]): List with the actions to convet
        mask(List[int]): List with the mask of the actions that
                will be deleted from the original list

    Returns:
        List[int]: Shorter list with the contents of x but
                with the fields that had a 0 in mask removed
    """
    
    ret: List[int] = list()
    for m,v in zip(mask,x):
        if m != 0:
            ret.append(v)
    return ret

def partial_action_2_full_action(x, mask):
    """
    Function that converts the list of actions x into
    a larger list given the mask

    Args:
        x(List[int]): List with the actions to convet
        mask(List[int]): List with the mask of the actions that
                will be added from the original list

    Returns:
        List[int]: Larger list with the contents of x but
                with the fields that had a 0 in mask added 
    """
    ret: List[int] = list()
    count:int = 0
    for idx,v in enumerate(mask):
        if v == 0:
            ret.append(0)
        else:
            ret.append(x[count])
            count += 1
    return ret
