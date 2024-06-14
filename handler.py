from model.yolo_model import Model


if __name__ == '__main__':
    model = Model()
    model.train()
    model.predict()
    model.auto_annotate()
