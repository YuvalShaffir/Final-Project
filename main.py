from data.AudioDataSet import CustomImageDataset
from training.Pipe import Pipe


def main():
    path_to_models_dir = ""
    pipe = Pipe(path_to_models_dir)
    pipe.run_pipe()


if __name__ == '__main__':
    main()

