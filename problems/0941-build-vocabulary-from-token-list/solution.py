def build_vocab(tokens):
    """
    Build a vocabulary dictionary from a list of tokens.

    Args:
        tokens: list of string tokens

    Returns:
        Dict mapping each unique token (sorted) to a unique integer ID starting from 0.
    """
    ans = {}
    for token in tokens:
        if token not in ans:
            ans[token]=1
        else:
            ans[token]+=1
    
    temp = list(ans.items())
    temp.sort(key=lambda item: item[0])
    ans = {}
    for i in range(len(temp)):
        ans[temp[i][0]]=i
    return ans
