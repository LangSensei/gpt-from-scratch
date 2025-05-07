from dataset import GptDataset

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

def test_dataloader_stride1():
    batch_size = 1
    max_length = 4
    stride = 1
    dataloader = GptDataset.create_dataloader(
        raw_text, batch_size=batch_size, max_length=max_length, stride=stride, shuffle=False
    )

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    assert inputs.shape == (batch_size, max_length), f"Expected shape ({batch_size}, {max_length}), got {inputs.shape}"
    assert targets.shape == (batch_size, max_length), f"Expected shape ({batch_size}, {max_length}), got {targets.shape}"
    assert targets[0][0] == inputs[0][1], f"Targets should be the inputs shifted by one position"

def test_dataloader_stride4():
    batch_size = 8
    max_length = 4
    stride = 4
    dataloader = GptDataset.create_dataloader(
        raw_text, batch_size=batch_size, max_length=max_length, stride=stride, shuffle=False
    )

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    assert inputs.shape == (batch_size, max_length), f"Expected shape ({batch_size}, {max_length}), got {inputs.shape}"
    assert targets.shape == (batch_size, max_length), f"Expected shape ({batch_size}, {max_length}), got {targets.shape}"
    assert targets[0][0] == inputs[0][1], f"Targets should be the inputs shifted by one position"