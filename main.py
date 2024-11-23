import argparse

from src.pipeline.train import train_and_evaluate, evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models.")
    parser.add_argument("--train", help="Train a model (e.g., --train xgb)", type=str)
    parser.add_argument("--evaluate", help="Evaluate a model (e.g., --evaluate xgb)", type=str)
    parser.add_argument("--save", help="Save the trained model", action="store_true")
    parser.add_argument("--version", help="Specify model version for evaluation", type=int)
    parser.add_argument("--verbose", help="Set verbosity level during training", type=int, default=1)

    args = parser.parse_args()

    if args.train:
        train_and_evaluate(args.train)
    elif args.evaluate:
        evaluate_model(args.evaluate)
    else:
        print("Error: Unsupported command. Use --train or --evaluate.")

if __name__ == "__main__":
    main()