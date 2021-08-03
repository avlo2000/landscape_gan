import config
from model import *
from data_loader import DataLoader
import torch
import pycuda.driver as cuda


def main():
    cuda.init()
    ## Get Id of default device
    dev = torch.cuda.current_device()
    print('GPU available: ' + str(torch.cuda.is_available()))
    dl = DataLoader(config.IMAGE_SIZES)
    dl.load()
    dl.prepare(transpose=True)
    model = AutoEncoder(config.IMAGE_SIZES)
    train_x, test_x, train_y, test_y = dl.get_data(0.8)
    train_model(train_x, test_x, train_y, test_y, model, 20, vis_test=False)


if __name__ == '__main__':
    main()
