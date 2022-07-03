import argparse
import fasttext

# fasttext directory
parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", help="Training file")
general.add_argument("--output", default="/workspace/datasets/fasttext/model.bin", help="the file to output to")
general.add_argument("--testdata", default="/workspace/datasets/fasttext/labeled_products_stemmed.test", help="the file to output to")
general.add_argument("--epoch", default=1, help="Number of epochs")
general.add_argument("--lr", default=0.1, help="Learning rate")
general.add_argument("--wordNgrams", default=1, help="Using ngrams")


args = parser.parse_args()
input_file = args.input
output_file = args.output
testdata = args.testdata
epoch = int(args.epoch)
lr = float(args.lr)
wordNgrams = int(args.wordNgrams)

if __name__ == '__main__':
    # Train model
    model = fasttext.train_supervised(input=input_file, epoch=epoch, lr=lr, wordNgrams=wordNgrams)
    model.save_model(output_file)
    model = fasttext.load_model(output_file)
    # Test model
    print(model.test(testdata))
