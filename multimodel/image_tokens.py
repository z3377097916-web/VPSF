"""Expand an atomic image token on CPU, exactly as repeating its string before tokenization."""
def packed_image_tokens(tokenizer, texts, marker_id, copies, torch):
    if copies < 1:
        raise ValueError('Empty image token span')
    data = tokenizer(texts, padding=True, return_tensors='pt')
    ids = data['input_ids']
    markers = ids.eq(marker_id)
    if not markers.sum(dim=1).eq(1).all():
        raise ValueError('Expected exactly one image marker in each tokenized prompt')
    if copies == 1:
        return data
    positions = markers.long().argmax(dim=1, keepdim=True)
    columns = torch.arange(ids.shape[1] + copies - 1)[None, :]
    gather = columns - (columns - positions).clamp(min=0, max=copies-1)
    for key, tensor in list(data.items()):
        if tensor.shape != ids.shape:
            raise ValueError(f'Unexpected tokenizer output layout for {key}')
        data[key] = tensor.gather(1, gather)
    return data
