import trading_data


if __name__ == "__main__":
    points = 5
    inputs, outputs = trading_data.DatasetGenerator.systhesis_play_operator_generator(points)
    inputs, outputs = trading_data.DatasetGenerator.systhesis_play_generator(points)
    nb_plays = 3
    inputs, outputs, plays_outputs = trading_data.DatasetGenerator.systhesis_model_generator(nb_plays, points, debug_plays=True)
    print(inputs.shape, outputs.shape, plays_outputs.shape)
