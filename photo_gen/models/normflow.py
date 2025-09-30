import torch
from torch.utils.data import TensorDataset, DataLoader
import hydra


def train(data: np.ndarray, cfg, checkpoint_path: os.PathLike, savedir: os.PathLike,
          run=None):
    # seed = -1
    # set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)
    n_epochs = cfg.n_epochs
    lr = cfg.lr
    batch_size = cfg.batch_size
    num_time_steps = cfg.num_time_steps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert data.shape[-2:] == cfg.image_shape
    print("TRAINING")
    print(f"{n_epochs} epochs total")
    dtype = torch.float32

    data = torch.tensor(data, dtype=dtype)
    if data.ndim == 3:
        data = data.unsqueeze(1)
    assert data.ndim == 4

    model = hydra.utils.instantiate(cfg.model)

    assert data.shape[-2:] == cfg.image_shape

    train_dataset = TensorDataset(data)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)