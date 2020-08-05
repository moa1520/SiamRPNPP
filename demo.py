import torch
from model import ModelBuilder
from tracker import build_tracker


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device :", device)

    # create model
    model = ModelBuilder()

    # load model
    checkpoint = torch.load("pretrained_model/SiamRPNres50.pth",
                            map_location=lambda storage, loc: storage.cpu())

    model.load_state_dict(checkpoint)
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)


if __name__ == "__main__":
    main()
