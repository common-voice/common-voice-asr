# PyTorch “Writing Custom Datasets, DataLoaders & Transforms”
- custom dataset inherits Dataset & overrides __len__ & __get_item__ 
- Dataloader provides features to batch, shuffle, & load data
    - collate_fn specifies how samples need to be batched - customizable

# librosa documentation for melspectrogram and power_to_db
melspectrogram
- librosa.feature.melspectrogram(*, y=None, sr=22050, S=None, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='constant', power=2.0, **kwargs)
power_to_db
- compute a mel-scaled spec