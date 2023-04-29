import argparse
import tensorflow as tf

def main(args):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=args.epochs)

    #model.save(f"{args.output_dir}/model")
    #tf.saved_model.save(model, f"{args.output_dir}/model")
    #model.save(f"{args.output_dir}/model", save_format='tf')
    #tf.saved_model.save(model, f"{args.output_dir}/model")
    #model.save(f"{args.output_dir}/model.h5")
    ##tf.saved_model.save(model, f"{args.output_dir}/model")
    tf.saved_model.save(model, "gs://29apr2023/output/model")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="Path to output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training")
    args = parser.parse_args()

    main(args)

