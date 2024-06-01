import utils.autoanchor as autoAC

if __name__ == '__main__':
    config = "./data/train_data.yaml"
    # 对数据集重新计算 anchors
    new_anchors = autoAC.kmean_anchors(config, 9, 640, 5.0, 1000, True)
    print(new_anchors)